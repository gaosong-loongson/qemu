/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * QEMU loongson 3a5000 develop board emulation
 *
 * Copyright (c) 2021 Loongson Technology Corporation Limited
 */
#include "qemu/osdep.h"
#include "qemu/units.h"
#include "qemu/datadir.h"
#include "qapi/error.h"
#include "hw/boards.h"
#include "hw/char/serial.h"
#include "sysemu/sysemu.h"
#include "sysemu/qtest.h"
#include "sysemu/runstate.h"
#include "sysemu/reset.h"
#include "sysemu/rtc.h"
#include "hw/loongarch/virt.h"
#include "exec/address-spaces.h"
#include "hw/irq.h"
#include "net/net.h"
#include "hw/loader.h"
#include "elf.h"
#include "hw/intc/loongarch_ipi.h"
#include "hw/intc/loongarch_extioi.h"
#include "hw/intc/loongarch_pch_pic.h"
#include "hw/intc/loongarch_pch_msi.h"
#include "hw/pci-host/ls7a.h"
#include "hw/pci-host/gpex.h"
#include "hw/misc/unimp.h"
#include "hw/loongarch/fw_cfg.h"
#include "target/loongarch/cpu.h"

#define PM_BASE 0x10080000
#define PM_SIZE 0x100
#define PM_CTRL 0x10

struct memmap_entry {
    uint64_t address;
    uint64_t length;
    uint32_t type;
    uint32_t reserved;
};

static struct memmap_entry *memmap_table;
static unsigned memmap_entries;

static int memmap_add_entry(uint64_t address, uint64_t length, uint32_t type)
{
    int i;

    for (i = 0; i < memmap_entries; i++) {
        if (memmap_table[i].address == address) {
            fprintf(stderr, "%s address:0x%lx length:0x%lx already exists\n",
                     __func__, address, length);
            return 0;
        }
    }

    memmap_table = g_renew(struct memmap_entry, memmap_table,
                                                      memmap_entries + 1);
    memmap_table[memmap_entries].address = cpu_to_le64(address);
    memmap_table[memmap_entries].length = cpu_to_le64(length);
    memmap_table[memmap_entries].type = cpu_to_le32(type);
    memmap_entries++;

    return memmap_entries;
}


/*
 * This is a placeholder for missing ACPI,
 * and will eventually be replaced.
 */
static uint64_t loongarch_virt_pm_read(void *opaque, hwaddr addr, unsigned size)
{
    return 0;
}

static void loongarch_virt_pm_write(void *opaque, hwaddr addr,
                               uint64_t val, unsigned size)
{
    if (addr != PM_CTRL) {
        return;
    }

    switch (val) {
    case 0x00:
        qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET);
        return;
    case 0xff:
        qemu_system_shutdown_request(SHUTDOWN_CAUSE_GUEST_SHUTDOWN);
        return;
    default:
        return;
    }
}

static const MemoryRegionOps loongarch_virt_pm_ops = {
    .read  = loongarch_virt_pm_read,
    .write = loongarch_virt_pm_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 1
    }
};

static struct _loaderparams {
    uint64_t ram_size;
    const char *kernel_filename;
    const char *kernel_cmdline;
    const char *initrd_filename;
} loaderparams;

static uint64_t cpu_loongarch_virt_to_phys(void *opaque, uint64_t addr)
{
    return addr & 0x1fffffffll;
}

static int64_t load_kernel_info(void)
{
    uint64_t kernel_entry, kernel_low, kernel_high;
    ssize_t kernel_size;

    kernel_size = load_elf(loaderparams.kernel_filename, NULL,
                           cpu_loongarch_virt_to_phys, NULL,
                           &kernel_entry, &kernel_low,
                           &kernel_high, NULL, 0,
                           EM_LOONGARCH, 1, 0);

    if (kernel_size < 0) {
        error_report("could not load kernel '%s': %s",
                     loaderparams.kernel_filename,
                     load_elf_strerror(kernel_size));
        exit(1);
    }
    return kernel_entry;
}

static void loongarch_devices_init(DeviceState *pch_pic)
{
    DeviceState *gpex_dev;
    SysBusDevice *d;
    PCIBus *pci_bus;
    MemoryRegion *ecam_alias, *ecam_reg, *pio_alias, *pio_reg;
    MemoryRegion *mmio_alias, *mmio_reg, *pm_mem;
    int i;

    gpex_dev = qdev_new(TYPE_GPEX_HOST);
    d = SYS_BUS_DEVICE(gpex_dev);
    sysbus_realize_and_unref(d, &error_fatal);
    pci_bus = PCI_HOST_BRIDGE(gpex_dev)->bus;

    /* Map only part size_ecam bytes of ECAM space */
    ecam_alias = g_new0(MemoryRegion, 1);
    ecam_reg = sysbus_mmio_get_region(d, 0);
    memory_region_init_alias(ecam_alias, OBJECT(gpex_dev), "pcie-ecam",
                             ecam_reg, 0, LS7A_PCI_CFG_SIZE);
    memory_region_add_subregion(get_system_memory(), LS7A_PCI_CFG_BASE,
                                ecam_alias);

    /* Map PCI mem space */
    mmio_alias = g_new0(MemoryRegion, 1);
    mmio_reg = sysbus_mmio_get_region(d, 1);
    memory_region_init_alias(mmio_alias, OBJECT(gpex_dev), "pcie-mmio",
                             mmio_reg, LS7A_PCI_MEM_BASE, LS7A_PCI_MEM_SIZE);
    memory_region_add_subregion(get_system_memory(), LS7A_PCI_MEM_BASE,
                                mmio_alias);

    /* Map PCI IO port space. */
    pio_alias = g_new0(MemoryRegion, 1);
    pio_reg = sysbus_mmio_get_region(d, 2);
    memory_region_init_alias(pio_alias, OBJECT(gpex_dev), "pcie-io", pio_reg,
                             LS7A_PCI_IO_OFFSET, LS7A_PCI_IO_SIZE);
    memory_region_add_subregion(get_system_memory(), LS7A_PCI_IO_BASE,
                                pio_alias);

    for (i = 0; i < GPEX_NUM_IRQS; i++) {
        sysbus_connect_irq(d, i,
                           qdev_get_gpio_in(pch_pic, 16 + i));
        gpex_set_irq_num(GPEX_HOST(gpex_dev), i, 16 + i);
    }

    serial_mm_init(get_system_memory(), LS7A_UART_BASE, 0,
                   qdev_get_gpio_in(pch_pic,
                                    LS7A_UART_IRQ - PCH_PIC_IRQ_OFFSET),
                   115200, serial_hd(0), DEVICE_LITTLE_ENDIAN);

    /* Network init */
    for (i = 0; i < nb_nics; i++) {
        NICInfo *nd = &nd_table[i];

        if (!nd->model) {
            nd->model = g_strdup("virtio");
        }

        pci_nic_init_nofail(nd, pci_bus, nd->model, NULL);
    }

    /* VGA setup */
    pci_vga_init(pci_bus);

    /*
     * There are some invalid guest memory access.
     * Create some unimplemented devices to emulate this.
     */
    create_unimplemented_device("pci-dma-cfg", 0x1001041c, 0x4);
    sysbus_create_simple("ls7a_rtc", LS7A_RTC_REG_BASE,
                         qdev_get_gpio_in(pch_pic,
                         LS7A_RTC_IRQ - PCH_PIC_IRQ_OFFSET));

    pm_mem = g_new(MemoryRegion, 1);
    memory_region_init_io(pm_mem, NULL, &loongarch_virt_pm_ops,
                          NULL, "loongarch_virt_pm", PM_SIZE);
    memory_region_add_subregion(get_system_memory(), PM_BASE, pm_mem);
}

static void loongarch_irq_init(LoongArchMachineState *lams)
{
    MachineState *ms = MACHINE(lams);
    DeviceState *pch_pic, *pch_msi, *cpudev;
    DeviceState *ipi, *extioi;
    SysBusDevice *d;
    LoongArchCPU *lacpu;
    CPULoongArchState *env;
    CPUState *cpu_state;
    int cpu, pin, i;

    ipi = qdev_new(TYPE_LOONGARCH_IPI);
    sysbus_realize_and_unref(SYS_BUS_DEVICE(ipi), &error_fatal);

    extioi = qdev_new(TYPE_LOONGARCH_EXTIOI);
    sysbus_realize_and_unref(SYS_BUS_DEVICE(extioi), &error_fatal);

    /*
     * The connection of interrupts:
     *   +-----+    +---------+     +-------+
     *   | IPI |--> | CPUINTC | <-- | Timer |
     *   +-----+    +---------+     +-------+
     *                  ^
     *                  |
     *            +---------+
     *            | EIOINTC |
     *            +---------+
     *             ^       ^
     *             |       |
     *      +---------+ +---------+
     *      | PCH-PIC | | PCH-MSI |
     *      +---------+ +---------+
     *        ^      ^          ^
     *        |      |          |
     * +--------+ +---------+ +---------+
     * | UARTs  | | Devices | | Devices |
     * +--------+ +---------+ +---------+
     */
    for (cpu = 0; cpu < ms->smp.cpus; cpu++) {
        cpu_state = qemu_get_cpu(cpu);
        cpudev = DEVICE(cpu_state);
        lacpu = LOONGARCH_CPU(cpu_state);
        env = &(lacpu->env);

        /* connect ipi irq to cpu irq */
        qdev_connect_gpio_out(ipi, cpu, qdev_get_gpio_in(cpudev, IRQ_IPI));
        /* IPI iocsr memory region */
        memory_region_add_subregion(&env->system_iocsr, SMP_IPI_MAILBOX,
                                    sysbus_mmio_get_region(SYS_BUS_DEVICE(ipi),
                                    cpu));
        /* extioi iocsr memory region */
        memory_region_add_subregion(&env->system_iocsr, APIC_BASE,
                                sysbus_mmio_get_region(SYS_BUS_DEVICE(extioi),
                                cpu));
    }

    /*
     * connect ext irq to the cpu irq
     * cpu_pin[9:2] <= intc_pin[7:0]
     */
    for (cpu = 0; cpu < ms->smp.cpus; cpu++) {
        cpudev = DEVICE(qemu_get_cpu(cpu));
        for (pin = 0; pin < LS3A_INTC_IP; pin++) {
            qdev_connect_gpio_out(extioi, (cpu * 8 + pin),
                                  qdev_get_gpio_in(cpudev, pin + 2));
        }
    }

    pch_pic = qdev_new(TYPE_LOONGARCH_PCH_PIC);
    d = SYS_BUS_DEVICE(pch_pic);
    sysbus_realize_and_unref(d, &error_fatal);
    memory_region_add_subregion(get_system_memory(), LS7A_IOAPIC_REG_BASE,
                            sysbus_mmio_get_region(d, 0));
    memory_region_add_subregion(get_system_memory(),
                            LS7A_IOAPIC_REG_BASE + PCH_PIC_ROUTE_ENTRY_OFFSET,
                            sysbus_mmio_get_region(d, 1));
    memory_region_add_subregion(get_system_memory(),
                            LS7A_IOAPIC_REG_BASE + PCH_PIC_INT_STATUS_LO,
                            sysbus_mmio_get_region(d, 2));

    /* Connect 64 pch_pic irqs to extioi */
    for (int i = 0; i < PCH_PIC_IRQ_NUM; i++) {
        qdev_connect_gpio_out(DEVICE(d), i, qdev_get_gpio_in(extioi, i));
    }

    pch_msi = qdev_new(TYPE_LOONGARCH_PCH_MSI);
    qdev_prop_set_uint32(pch_msi, "msi_irq_base", PCH_MSI_IRQ_START);
    d = SYS_BUS_DEVICE(pch_msi);
    sysbus_realize_and_unref(d, &error_fatal);
    sysbus_mmio_map(d, 0, LS7A_PCH_MSI_ADDR_LOW);
    for (i = 0; i < PCH_MSI_IRQ_NUM; i++) {
        /* Connect 192 pch_msi irqs to extioi */
        qdev_connect_gpio_out(DEVICE(d), i,
                              qdev_get_gpio_in(extioi, i + PCH_MSI_IRQ_START));
    }

    loongarch_devices_init(pch_pic);
}

static void loongarch_firmware_init(LoongArchMachineState *lams)
{
    char *filename = MACHINE(lams)->firmware;
    char *bios_name = NULL;
    int bios_size;

    lams->bios_loaded = false;
    if (filename) {
        bios_name = qemu_find_file(QEMU_FILE_TYPE_BIOS, filename);
        if (!bios_name) {
            error_report("Could not find ROM image '%s'", filename);
            exit(1);
        }

        bios_size = load_image_targphys(bios_name, VIRT_BIOS_BASE, VIRT_BIOS_SIZE);
        if (bios_size < 0) {
            error_report("Could not load ROM image '%s'", bios_name);
            exit(1);
        }

        g_free(bios_name);

        memory_region_init_ram(&lams->bios, NULL, "loongarch.bios",
                               VIRT_BIOS_SIZE, &error_fatal);
        memory_region_set_readonly(&lams->bios, true);
        memory_region_add_subregion(get_system_memory(), VIRT_BIOS_BASE, &lams->bios);
        lams->bios_loaded = true;
    }

}

static void reset_load_elf(void *opaque)
{
    LoongArchCPU *cpu = opaque;
    CPULoongArchState *env = &cpu->env;

    cpu_reset(CPU(cpu));
    if (env->load_elf) {
        cpu_set_pc(CPU(cpu), env->elf_address);
    }
}

/* Load an image file into an fw_cfg entry identified by key. */
static void load_image_to_fw_cfg(FWCfgState *fw_cfg, uint16_t size_key,
                                 uint16_t data_key, const char *image_name,
                                 bool try_decompress)
{
    size_t size = -1;
    uint8_t *data;

    if (image_name == NULL) {
        return;
    }

    if (try_decompress) {
        size = load_image_gzipped_buffer(image_name,
                                         LOAD_IMAGE_MAX_GUNZIP_BYTES, &data);
    }

    if (size == (size_t)-1) {
        gchar *contents;
        gsize length;

        if (!g_file_get_contents(image_name, &contents, &length, NULL)) {
            error_report("failed to load \"%s\"", image_name);
            exit(1);
        }
        size = length;
        data = (uint8_t *)contents;
    }

    fw_cfg_add_i32(fw_cfg, size_key, size);
    fw_cfg_add_bytes(fw_cfg, data_key, data, size);
}

static void fw_cfg_add_kernel_info(FWCfgState *fw_cfg)
{
    /*
     * Expose the kernel, the command line, and the initrd in fw_cfg.
     * We don't process them here at all, it's all left to the
     * firmware.
     */
    load_image_to_fw_cfg(fw_cfg,
                         FW_CFG_KERNEL_SIZE, FW_CFG_KERNEL_DATA,
                         loaderparams.kernel_filename,
                         false);

    if (loaderparams.initrd_filename) {
        load_image_to_fw_cfg(fw_cfg,
                             FW_CFG_INITRD_SIZE, FW_CFG_INITRD_DATA,
                             loaderparams.initrd_filename, false);
    }

    if (loaderparams.kernel_cmdline) {
        fw_cfg_add_i32(fw_cfg, FW_CFG_CMDLINE_SIZE,
                       strlen(loaderparams.kernel_cmdline) + 1);
        fw_cfg_add_string(fw_cfg, FW_CFG_CMDLINE_DATA,
                          loaderparams.kernel_cmdline);
    }
}

static void loongarch_firmware_boot(LoongArchMachineState *lams)
{
    fw_cfg_add_kernel_info(lams->fw_cfg);
}

static void loongarch_direct_kernel_boot(LoongArchMachineState *lams)
{
    MachineState *machine = MACHINE(lams);
    int64_t kernel_addr = 0;
    LoongArchCPU *lacpu;
    int i;

    kernel_addr = load_kernel_info();
    if (!machine->firmware) {
        for (i = 0; i < machine->smp.cpus; i++) {
            lacpu = LOONGARCH_CPU(qemu_get_cpu(i));
            lacpu->env.load_elf = true;
            lacpu->env.elf_address = kernel_addr;
        }
    }
}

static void loongarch_init(MachineState *machine)
{
    LoongArchCPU *lacpu;
    const char *cpu_model = machine->cpu_type;
    ram_addr_t offset = 0;
    ram_addr_t ram_size = machine->ram_size;
    uint64_t highram_size = 0;
    MemoryRegion *address_space_mem = get_system_memory();
    LoongArchMachineState *lams = LOONGARCH_MACHINE(machine);
    int i;

    if (!cpu_model) {
        cpu_model = LOONGARCH_CPU_TYPE_NAME("la464");
    }

    if (!strstr(cpu_model, "la464")) {
        error_report("LoongArch/TCG needs cpu type la464");
        exit(1);
    }

    if (ram_size < 1 * GiB) {
        error_report("ram_size must be greater than 1G.");
        exit(1);
    }

    /* Init CPUs */
    for (i = 0; i < machine->smp.cpus; i++) {
        cpu_create(machine->cpu_type);
    }

    /* Add memory region */
    memory_region_init_alias(&lams->lowmem, NULL, "loongarch.lowram",
                             machine->ram, 0, 256 * MiB);
    memory_region_add_subregion(address_space_mem, offset, &lams->lowmem);
    offset += 256 * MiB;
    memmap_add_entry(0, 256 * MiB, 1);
    highram_size = ram_size - 256 * MiB;
    memory_region_init_alias(&lams->highmem, NULL, "loongarch.highmem",
                             machine->ram, offset, highram_size);
    memory_region_add_subregion(address_space_mem, 0x90000000, &lams->highmem);
    memmap_add_entry(0x90000000, highram_size, 1);
    /* Add isa io region */
    memory_region_init_alias(&lams->isa_io, NULL, "isa-io",
                             get_system_io(), 0, LOONGARCH_ISA_IO_SIZE);
    memory_region_add_subregion(address_space_mem, LOONGARCH_ISA_IO_BASE,
                                &lams->isa_io);
    /* load the BIOS image. */
    loongarch_firmware_init(lams);

    /* fw_cfg init */
    lams->fw_cfg = loongarch_fw_cfg_init(ram_size, machine);
    rom_set_fw(lams->fw_cfg);

    if (lams->fw_cfg != NULL) {
        fw_cfg_add_file(lams->fw_cfg, "etc/memmap",
                        memmap_table,
                        sizeof(struct memmap_entry) * (memmap_entries));
    }
    loaderparams.ram_size = ram_size;
    loaderparams.kernel_filename = machine->kernel_filename;
    loaderparams.kernel_cmdline = machine->kernel_cmdline;
    loaderparams.initrd_filename = machine->initrd_filename;
    /* load the kernel. */
    if (loaderparams.kernel_filename) {
        if (lams->bios_loaded) {
            loongarch_firmware_boot(lams);
        } else {
            loongarch_direct_kernel_boot(lams);
        }
    }
    /* register reset function */
    for (i = 0; i < machine->smp.cpus; i++) {
        lacpu = LOONGARCH_CPU(qemu_get_cpu(i));
        qemu_register_reset(reset_load_elf, lacpu);
    }
    /* Initialize the IO interrupt subsystem */
    loongarch_irq_init(lams);
}

static void loongarch_class_init(ObjectClass *oc, void *data)
{
    MachineClass *mc = MACHINE_CLASS(oc);

    mc->desc = "Loongson-3A5000 LS7A1000 machine";
    mc->init = loongarch_init;
    mc->default_ram_size = 1 * GiB;
    mc->default_cpu_type = LOONGARCH_CPU_TYPE_NAME("la464");
    mc->default_ram_id = "loongarch.ram";
    mc->max_cpus = LOONGARCH_MAX_VCPUS;
    mc->is_default = 1;
    mc->default_kernel_irqchip_split = false;
    mc->block_default_type = IF_VIRTIO;
    mc->default_boot_order = "c";
    mc->no_cdrom = 1;
}

static const TypeInfo loongarch_machine_types[] = {
    {
        .name           = TYPE_LOONGARCH_MACHINE,
        .parent         = TYPE_MACHINE,
        .instance_size  = sizeof(LoongArchMachineState),
        .class_init     = loongarch_class_init,
    }
};

DEFINE_TYPES(loongarch_machine_types)
