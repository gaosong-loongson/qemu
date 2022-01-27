loongson3-ls7a generic platform (``loongson3-ls7a``)
====================================================

Use ``loongson3-ls7a`` machine type to emulate the loongson7a board.
There are some devices on loongson7a board, such as RTC device,
IOAPIC device, ACPI device and so on.

Supported devices
-----------------

The ``loongson3-ls7a`` machine supports:
- PCI/PCIe devices
- Memory device
- CPU device
- Ls7a host bridge
- Ls7a RTC device
- Ls7a IOAPIC device
- Ls7a ACPI device
- Fw_cfg device
- CPU device. Type: Loongson-3A5000.

CPU and machine Type
--------------------

The ``qemu-system-loongarch64`` provides emulation for loongson7a
machine. You can specify the machine type ``loongson3-ls7a`` and
cpu type ``Loongson-3A5000``.

Boot options
------------

Now the ``loongson3-ls7a`` machine can start using -bios parameter:

.. code-block:: bash

  $ qemu-system-loongarch64 -M loongson3-ls7a -cpu Loongson-3A5000 -smp 2 -m 1G \
      -display none -serial stdio \
      -bios loongarch_bios.bini

Info mtree
----------
address-space: memory
  0000000000000000-ffffffffffffffff (prio 0, i/o): system
    0000000000000000-ffffffffffffffff (prio 0, i/o): ls7a_mmio
      00000000000a0000-00000000000bffff (prio 1, i/o): cirrus-lowmem-container
        00000000000a0000-00000000000bffff (prio 0, i/o): cirrus-low-memory
    0000000000000000-000000000fffffff (prio 0, ram): alias loongarch.lowram @loongarch.ram 0000000000000000-000000000fffffff
    0000000010000000-00000000100000ff (prio 0, i/o): loongarch_pch_pic.reg32_part1
    0000000010000100-000000001000039f (prio 0, i/o): loongarch_pch_pic.reg8
    00000000100003a0-0000000010000fff (prio 0, i/o): loongarch_pch_pic.reg32_part2
    000000001001041c-000000001001041f (prio -1000, i/o): pci-dma-cfg
    0000000010013ffc-0000000010013fff (prio -1000, i/o): mmio fallback 1
    00000000100d0000-00000000100d00ff (prio 0, i/o): ls7a_pm
      00000000100d000c-00000000100d0013 (prio 0, i/o): acpi-evt
      00000000100d0014-00000000100d0017 (prio 0, i/o): acpi-cnt
      00000000100d0018-00000000100d001b (prio 0, i/o): acpi-tmr
      00000000100d0028-00000000100d002f (prio 0, i/o): acpi-gpe0
      00000000100d0030-00000000100d0033 (prio 0, i/o): acpi-reset
    00000000100d0100-00000000100d01ff (prio 0, i/o): ls7a_rtc
    0000000018000000-0000000018003fff (prio 0, i/o): alias isa-io @io 0000000000000000-0000000000003fff
    0000000018004000-000000001800ffff (prio 0, i/o): alias ls7a-pci-io @io 0000000000004000-000000000000ffff
    000000001a000000-000000001bffffff (prio 0, i/o): ls7a_pci_conf
    000000001c000000-000000001c3fffff (prio 0, rom): loongarch.bios
    000000001e020000-000000001e020001 (prio 0, i/o): fwcfg.ctl
    000000001e020008-000000001e02000f (prio 0, i/o): fwcfg.data
    000000001fe001e0-000000001fe001e7 (prio 0, i/o): serial
    0000000020000000-0000000027ffffff (prio 0, i/o): pcie-mmcfg-mmio
    000000002ff00000-000000002ff00007 (prio 0, i/o): loongarch_pch_msi
    0000000090000000-000000017fffffff (prio 0, ram): alias loongarch.highmem @loongarch.ram 0000000010000000-00000000ffffffff

address-space: IOCSR
  0000000000000000-ffffffffffffffff (prio 0, i/o): iocsr
    0000000000000008-0000000000000427 (prio 0, i/o): iocsr_misc
    0000000000001000-00000000000010ff (prio 0, i/o): loongarch_ipi
    0000000000001400-00000000000014bf (prio 0, i/o): loongarch_extioi.nodetype
    00000000000014c0-000000000000167f (prio 0, i/o): loongarch_extioi.ipmap_enable
    0000000000001680-0000000000001bff (prio 0, i/o): loongarch_extioi.bounce_coreisr
    0000000000001c00-0000000000001cff (prio 0, i/o): loongarch_extioi.coremap
