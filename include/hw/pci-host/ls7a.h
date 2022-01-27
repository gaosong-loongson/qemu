/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * QEMU LoongArch CPU
 *
 * Copyright (c) 2021 Loongson Technology Corporation Limited
 */

#ifndef HW_LS7A_H
#define HW_LS7A_H

#include "hw/pci/pci.h"
#include "hw/pci/pcie_host.h"
#include "hw/pci-host/pam.h"
#include "qemu/units.h"
#include "qemu/range.h"
#include "qom/object.h"

#define HT1LO_PCICFG_BASE        0x1a000000
#define HT1LO_PCICFG_SIZE        0x02000000

#define LS_PCIECFG_BASE          0x20000000
#define LS_PCIECFG_SIZE          0x08000000

#define LS7A_PCI_IO_BASE         0x18004000UL
#define LS7A_PCI_IO_SIZE         0xC000
#define LS7A_PCI_IO_OFFSET       0x4000

#define LS7A_PCH_REG_BASE       0x10000000UL
#define LS7A_IOAPIC_REG_BASE    (LS7A_PCH_REG_BASE)
#define LS7A_PCH_MSI_ADDR_LOW   0x2FF00000UL

/*
 * According to the kernel pch irq start from 64 offset
 * 0 ~ 16 irqs used for non-pci device while 16 ~ 64 irqs
 * used for pci device.
 */
#define PCH_PIC_IRQ_OFFSET      64
#define LS7A_DEVICE_IRQS        16
#define LS7A_PCI_IRQS           48

#define LS7A_UART_IRQ           (PCH_PIC_IRQ_OFFSET + 2)
#define LS7A_UART_BASE          0x1fe001e0
#define LS7A_RTC_IRQ            (PCH_PIC_IRQ_OFFSET + 3)
#define LS7A_MISC_REG_BASE      (LS7A_PCH_REG_BASE + 0x00080000)
#define LS7A_RTC_REG_BASE       (LS7A_MISC_REG_BASE + 0x00050100)
#define LS7A_RTC_LEN            0x100

struct LS7APCIState {
    /*< private >*/
    PCIDevice parent_obj;
    /*< public >*/
};

typedef struct LS7APCIState LS7APCIState;
typedef struct LS7APCIEHost {
    /*< private >*/
    PCIExpressHost parent_obj;
    /*< public >*/

    LS7APCIState pci_dev;

    qemu_irq irqs[LS7A_PCI_IRQS];
    MemoryRegion pci_conf;
    MemoryRegion pci_mmio;
    MemoryRegion pci_io;
} LS7APCIEHost;

#define TYPE_LS7A_HOST_DEVICE "ls7a1000-pciehost"
OBJECT_DECLARE_SIMPLE_TYPE(LS7APCIEHost, LS7A_HOST_DEVICE)

#define TYPE_LS7A_PCI_DEVICE "ls7a1000_pci"
OBJECT_DECLARE_SIMPLE_TYPE(LS7APCIState, LS7A_PCI_DEVICE)

#endif /* HW_LS7A_H */
