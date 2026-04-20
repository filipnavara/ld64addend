.text
.align 2
.globl _main
_main:
    mov x0, 0
    ret

.section __TEXT,__bar
.globl _bar
_bar:

.section __TEXT,__foo
.globl _foo
_foo:
    .rept 0x40001
    .long 0
    .endr
ltmp0:
    .rept 0x140002 //- 0x40001
    .long _bar - .
    .endr
