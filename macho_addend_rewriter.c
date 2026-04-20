/*
 * macho_addend_rewriter.c — Mach-O object file post-processor
 *
 * PROBLEM
 * -------
 * Newer versions of the Xcode linker (ld, not ld_classic) contain an internal
 * assertion in Fixup.cpp that fires when an object file has too many unique
 * large addends in SUBTRACTOR+UNSIGNED relocation pairs:
 *
 *     _addend == uniqueIndex && "too many large addends"
 *
 * This happens when the compiler emits difference-of-symbols relocations
 * (e.g. `.long _bar - .`) that all reference a single local section symbol
 * far from the relocation site.  As the distance grows the inline addend
 * stored in the section data grows proportionally and eventually overflows
 * the linker's internal unique-addend table.
 *
 * SOLUTION
 * --------
 * We insert synthetic local section symbols (named laddend$NNNNNN) at
 * strategic positions so that each SUBTRACTOR relocation can reference a
 * nearby symbol instead of a distant one.  This keeps all addends within a
 * configurable radius (default 3.5 MiB).  Earlier experiments suggested that
 * 4 MiB was sufficient, but larger real-world objects still tripped the
 * linker's unique-addend table near that boundary.
 *
 * The number of synthetic symbols is minimised with a greedy interval-
 * covering algorithm: each relocation site defines an interval [low, high]
 * of acceptable label addresses, and we greedily place labels at safe
 * relocation boundaries (rather than arbitrary byte offsets) so the added
 * symbols do not split an atom in the middle of a relocated field.
 *
 * The rewrite is semantics-preserving: only the subtractor symbol choice
 * and the corresponding inline addend change; the linked result is
 * identical.
 *
 * USAGE
 *     clang -c input.S -o input.o
 *     macho_addend_rewriter input.o        # rewrite in place
 *     clang input.o -o output              # link normally
 *
 * Supports both arm64 and x86_64 Mach-O object files.
 */

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <mach-o/arm64/reloc.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <mach-o/reloc.h>
#include <mach-o/x86_64/reloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* Extracted section metadata (1-based index matches Mach-O n_sect convention). */
typedef struct {
    uint32_t index;      /* 1-based section ordinal */
    char name[17];       /* null-terminated section name (max 16 chars) */
    uint64_t addr;       /* virtual address of the section start */
    uint64_t size;
    uint32_t offset;     /* file offset of the section data */
    uint32_t reloff;     /* file offset of the relocation entries */
    uint32_t nreloc;     /* number of relocation entries */
} SectionInfo;

/* Subset of nlist_64 fields we need for planning. */
typedef struct {
    uint32_t index;      /* original symbol table index */
    uint8_t n_type;
    uint8_t n_sect;      /* 1-based section ordinal, or 0 for absolute */
    uint64_t n_value;    /* symbol address */
} SymbolInfo;

/*
 * Decoded fields from the second 32-bit word of a relocation_info entry.
 * Mach-O packs these into a bitfield: [23:0] symbolnum, [24] pcrel,
 * [26:25] length (0=byte,1=word,2=long,3=quad), [27] extern, [31:28] type.
 */
typedef struct {
    uint32_t symbolnum;
    uint8_t pcrel;
    uint8_t length;
    uint8_t is_extern;
    uint8_t reloc_type;
} RelocFields;

/*
 * A relocation pair whose addend exceeds the safe threshold and therefore
 * needs to be rewritten to use a nearer local symbol.
 *
 * ideal_addr is where the replacement symbol would make the addend exactly
 * zero.  [low_addr, high_addr] is the interval of symbol addresses that
 * keep the addend within ±max_addend.
 */
typedef struct {
    uint32_t section_index;       /* 1-based */
    uint64_t reloc_addr;          /* section-relative offset of the pair */
    uint64_t site_addr;           /* absolute address of the relocated field */
    uint32_t reloc_entry_index;   /* index into the section's reloc table */
    uint32_t subtractor_symbol;   /* original subtractor symbol index */
    uint32_t unsigned_symbol;     /* original unsigned symbol index */
    uint8_t length;               /* relocation width code (0–3) */
    int64_t old_symbol_addr;      /* n_value of the original subtractor sym */
    int64_t addend;               /* inline addend read from section data */
    int64_t ideal_addr;           /* address that would make addend == 0 */
    int64_t low_addr;             /* lowest acceptable replacement sym addr */
    int64_t high_addr;            /* highest acceptable replacement sym addr */
} Candidate;

/* Identifies a synthetic label by its section and virtual address. */
typedef struct {
    uint32_t section_index;  /* 1-based */
    uint64_t addr;
} LabelKey;

/*
 * Records the decision for one candidate: either reuse an existing local
 * symbol (uses_existing_symbol == true, symbol_index set) or create a new
 * synthetic label (uses_existing_symbol == false, label_key set).
 */
typedef struct {
    Candidate candidate;
    bool uses_existing_symbol;
    uint32_t symbol_index;   /* valid when uses_existing_symbol */
    LabelKey label_key;      /* valid when !uses_existing_symbol */
} Assignment;

typedef struct {
    uint32_t *values;
    size_t count;
    size_t capacity;
} UInt32Array;

typedef struct {
    uint64_t *values;
    size_t count;
    size_t capacity;
} UInt64Array;

typedef struct {
    Candidate *values;
    size_t count;
    size_t capacity;
} CandidateArray;

typedef struct {
    Assignment *values;
    size_t count;
    size_t capacity;
} AssignmentArray;

typedef struct {
    LabelKey *values;
    size_t count;
    size_t capacity;
} LabelArray;

/* Parsed representation of a 64-bit Mach-O object file (in-memory). */
typedef struct {
    uint8_t *data;                          /* raw file bytes (mutable via COW) */
    size_t size;
    bool data_is_mmapped;                   /* true → munmap, false → free */
    cpu_type_t cputype;
    SectionInfo *sections;
    size_t section_count;
    size_t section_capacity;
    SymbolInfo *symbols;
    size_t symbol_count;
    UInt32Array *local_symbols_by_section;   /* index 0 unused; [n_sect] → list of local sym indices */
    uint32_t symtab_cmd_offset;              /* file offset of LC_SYMTAB load command */
    uint32_t dysymtab_cmd_offset;            /* file offset of LC_DYSYMTAB load command */
    uint8_t reloc_unsigned_type;             /* arch-specific UNSIGNED reloc type */
    uint8_t reloc_subtractor_type;           /* arch-specific SUBTRACTOR reloc type */
    const char *arch_name;
    struct symtab_command symtab;
    struct dysymtab_command dysymtab;
} MachOObject;

typedef struct {
    AssignmentArray assignments;
    LabelArray new_labels;
    size_t rewritten_relocations;
} RewritePlan;

/* Context passed to helpers that search for existing local symbols. */
typedef struct {
    const SymbolInfo *symbols;
    const UInt32Array *local_symbols;
    uint32_t section_index;   /* 1-based */
} GroupContext;

/* --- Dynamic array helpers (generic growth + type-safe append wrappers) --- */

static void free_uint32_array(UInt32Array *array) {
    free(array->values);
    array->values = NULL;
    array->count = 0;
    array->capacity = 0;
}

static void free_uint64_array(UInt64Array *array) {
    free(array->values);
    array->values = NULL;
    array->count = 0;
    array->capacity = 0;
}

static void free_candidate_array(CandidateArray *array) {
    free(array->values);
    array->values = NULL;
    array->count = 0;
    array->capacity = 0;
}

static void free_assignment_array(AssignmentArray *array) {
    free(array->values);
    array->values = NULL;
    array->count = 0;
    array->capacity = 0;
}

static void free_label_array(LabelArray *array) {
    free(array->values);
    array->values = NULL;
    array->count = 0;
    array->capacity = 0;
}

static void free_macho_object(MachOObject *object) {
    if (object == NULL) {
        return;
    }

    if (object->local_symbols_by_section != NULL) {
        for (size_t i = 0; i <= object->section_count; ++i) {
            free_uint32_array(&object->local_symbols_by_section[i]);
        }
    }

    free(object->local_symbols_by_section);
    free(object->symbols);
    free(object->sections);
    if (object->data_is_mmapped) {
        munmap(object->data, object->size);
    } else {
        free(object->data);
    }
    memset(object, 0, sizeof(*object));
}

static void free_rewrite_plan(RewritePlan *plan) {
    if (plan == NULL) {
        return;
    }
    free_assignment_array(&plan->assignments);
    free_label_array(&plan->new_labels);
    memset(plan, 0, sizeof(*plan));
}

/*
 * Ensure *buffer has room for at least `needed` elements, each of
 * `element_size` bytes.  Doubles the capacity until sufficient.
 */
static int reserve_bytes(void **buffer, size_t element_size, size_t *capacity, size_t needed) {
    if (needed <= *capacity) {
        return 0;
    }

    size_t new_capacity = *capacity == 0 ? 16 : *capacity;
    while (new_capacity < needed) {
        if (new_capacity > SIZE_MAX / 2) {
            return -1;
        }
        new_capacity *= 2;
    }

    void *new_buffer = realloc(*buffer, new_capacity * element_size);
    if (new_buffer == NULL) {
        return -1;
    }

    *buffer = new_buffer;
    *capacity = new_capacity;
    return 0;
}

static int append_uint32(UInt32Array *array, uint32_t value) {
    if (reserve_bytes((void **)&array->values, sizeof(uint32_t), &array->capacity, array->count + 1) != 0) {
        return -1;
    }
    array->values[array->count++] = value;
    return 0;
}

static int append_uint64(UInt64Array *array, uint64_t value) {
    if (reserve_bytes((void **)&array->values, sizeof(uint64_t), &array->capacity, array->count + 1) != 0) {
        return -1;
    }
    array->values[array->count++] = value;
    return 0;
}

static int compare_uint64(const void *lhs, const void *rhs) {
    const uint64_t a = *(const uint64_t *)lhs;
    const uint64_t b = *(const uint64_t *)rhs;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}

static void deduplicate_uint64_array(UInt64Array *array) {
    if (array->count < 2) {
        return;
    }

    size_t write_index = 1;
    for (size_t read_index = 1; read_index < array->count; ++read_index) {
        if (array->values[read_index] != array->values[write_index - 1]) {
            array->values[write_index++] = array->values[read_index];
        }
    }
    array->count = write_index;
}

static int append_candidate(CandidateArray *array, const Candidate *value) {
    if (reserve_bytes((void **)&array->values, sizeof(Candidate), &array->capacity, array->count + 1) != 0) {
        return -1;
    }
    array->values[array->count++] = *value;
    return 0;
}

static int append_assignment(AssignmentArray *array, const Assignment *value) {
    if (reserve_bytes((void **)&array->values, sizeof(Assignment), &array->capacity, array->count + 1) != 0) {
        return -1;
    }
    array->values[array->count++] = *value;
    return 0;
}

static int append_label(LabelArray *array, const LabelKey *value) {
    if (reserve_bytes((void **)&array->values, sizeof(LabelKey), &array->capacity, array->count + 1) != 0) {
        return -1;
    }
    array->values[array->count++] = *value;
    return 0;
}

/* Append a label only if no entry with the same (section, addr) exists. */
static int append_label_unique(LabelArray *array, uint32_t section_index, uint64_t addr) {
    for (size_t i = 0; i < array->count; ++i) {
        if (array->values[i].section_index == section_index && array->values[i].addr == addr) {
            return 0;
        }
    }

    LabelKey key = {.section_index = section_index, .addr = addr};
    return append_label(array, &key);
}

static int fail(const char *message) {
    fprintf(stderr, "error: %s\n", message);
    return 1;
}

static int fail_errno(const char *context) {
    fprintf(stderr, "error: %s: %s\n", context, strerror(errno));
    return 1;
}

/* --- Relocation bitfield encoding/decoding ---
 *
 * A Mach-O relocation_info is two 32-bit words: r_address and a packed word.
 * The packed word layout (little-endian):
 *   bits [23: 0] — r_symbolnum (symbol table index or section ordinal)
 *   bit  [24]    — r_pcrel
 *   bits [26:25] — r_length (0=byte, 1=word, 2=long, 3=quad)
 *   bit  [27]    — r_extern
 *   bits [31:28] — r_type (architecture-specific)
 */

static void decode_reloc_fields(uint32_t raw_word, RelocFields *fields) {
    fields->symbolnum = raw_word & 0x00FFFFFFu;
    fields->pcrel = (uint8_t)((raw_word >> 24) & 0x1u);
    fields->length = (uint8_t)((raw_word >> 25) & 0x3u);
    fields->is_extern = (uint8_t)((raw_word >> 27) & 0x1u);
    fields->reloc_type = (uint8_t)((raw_word >> 28) & 0xFu);
}

/* Replace only the symbolnum (low 24 bits) in a packed relocation word. */
static uint32_t encode_reloc_symbol(uint32_t raw_word, uint32_t symbol_index) {
    return (raw_word & 0xFF000000u) | symbol_index;
}

/* True if the symbol is a non-external, section-defined local (N_SECT). */
static bool symbol_is_local_section(const SymbolInfo *symbol) {
    return (symbol->n_type & N_EXT) == 0 && (symbol->n_type & N_TYPE) == N_SECT;
}

static int64_t section_end_addr(const SectionInfo *section) {
    return (int64_t)(section->addr + section->size);
}

/*
 * Read the inline signed addend at a relocation site.
 * r_address (reloc_addr) is section-relative, so the file offset is
 * section->offset + reloc_addr.  The width is determined by the
 * relocation's length field (0=1B, 1=2B, 2=4B, 3=8B).
 */
static int read_signed_addend(const MachOObject *object, const SectionInfo *section, uint64_t reloc_addr, uint8_t length, int64_t *value_out) {
    uint64_t data_offset = (uint64_t)section->offset + reloc_addr;
    if (data_offset >= object->size) {
        return fail("relocation points outside the section data");
    }

    switch (length) {
        case 0: {
            int8_t value;
            memcpy(&value, object->data + data_offset, sizeof(value));
            *value_out = value;
            return 0;
        }
        case 1: {
            int16_t value;
            if (data_offset + sizeof(value) > object->size) {
                return fail("relocation points outside the file");
            }
            memcpy(&value, object->data + data_offset, sizeof(value));
            *value_out = value;
            return 0;
        }
        case 2: {
            int32_t value;
            if (data_offset + sizeof(value) > object->size) {
                return fail("relocation points outside the file");
            }
            memcpy(&value, object->data + data_offset, sizeof(value));
            *value_out = value;
            return 0;
        }
        case 3: {
            int64_t value;
            if (data_offset + sizeof(value) > object->size) {
                return fail("relocation points outside the file");
            }
            memcpy(&value, object->data + data_offset, sizeof(value));
            *value_out = value;
            return 0;
        }
        default:
            return fail("unsupported relocation width");
    }
}

/* Write a new signed addend back into the section data. */
static int write_signed_addend(MachOObject *object, const SectionInfo *section, uint64_t reloc_addr, uint8_t length, int64_t value) {
    uint64_t data_offset = (uint64_t)section->offset + reloc_addr;
    if (data_offset >= object->size) {
        return fail("relocation points outside the section data");
    }

    switch (length) {
        case 0: {
            if (value < INT8_MIN || value > INT8_MAX) {
                return fail("rewritten addend does not fit in 8 bits");
            }
            int8_t written = (int8_t)value;
            memcpy(object->data + data_offset, &written, sizeof(written));
            return 0;
        }
        case 1: {
            if (data_offset + sizeof(int16_t) > object->size) {
                return fail("relocation points outside the file");
            }
            if (value < INT16_MIN || value > INT16_MAX) {
                return fail("rewritten addend does not fit in 16 bits");
            }
            int16_t written = (int16_t)value;
            memcpy(object->data + data_offset, &written, sizeof(written));
            return 0;
        }
        case 2: {
            if (data_offset + sizeof(int32_t) > object->size) {
                return fail("relocation points outside the file");
            }
            if (value < INT32_MIN || value > INT32_MAX) {
                return fail("rewritten addend does not fit in 32 bits");
            }
            int32_t written = (int32_t)value;
            memcpy(object->data + data_offset, &written, sizeof(written));
            return 0;
        }
        case 3: {
            if (data_offset + sizeof(int64_t) > object->size) {
                return fail("relocation points outside the file");
            }
            int64_t written = value;
            memcpy(object->data + data_offset, &written, sizeof(written));
            return 0;
        }
        default:
            return fail("unsupported relocation width");
    }
}

static int64_t parse_integer(const char *text, const char *flag_name, bool *ok) {
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > INT64_MAX) {
        fprintf(stderr, "error: invalid value for %s: %s\n", flag_name, text);
        *ok = false;
        return 0;
    }
    *ok = true;
    return (int64_t)parsed;
}

/*
 * Memory-map the file at `path` with MAP_PRIVATE (copy-on-write).
 * This avoids reading the entire file upfront; the OS pages in data on
 * demand.  Writes go to private pages and never touch the underlying file.
 */
static int mmap_file(const char *path, uint8_t **data_out, size_t *size_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return fail_errno(path);
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return fail_errno("fstat");
    }

    if (st.st_size <= 0) {
        close(fd);
        return fail("input file is empty or has an invalid size");
    }

    size_t size = (size_t)st.st_size;
    void *mapping = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapping == MAP_FAILED) {
        return fail_errno("mmap");
    }

    *data_out = mapping;
    *size_out = size;
    return 0;
}

/* Bounds-check: ensure [offset, offset+length) lies within [0, size). */
static int check_range(size_t size, size_t offset, size_t length, const char *what) {
    if (offset > size || length > size - offset) {
        fprintf(stderr, "error: %s is truncated\n", what);
        return 1;
    }
    return 0;
}

/*
 * Parse a 64-bit Mach-O object file from raw bytes into `object`.
 *
 * Extracts:
 *   - All sections (from LC_SEGMENT_64 commands)
 *   - LC_SYMTAB / LC_DYSYMTAB metadata
 *   - All symbols, indexed; local section symbols bucketed by n_sect
 *   - Architecture-specific relocation type constants (arm64 / x86_64)
 */
static int parse_macho_object(uint8_t *data, size_t size, bool data_is_mmapped, MachOObject *object) {
    memset(object, 0, sizeof(*object));
    object->data = data;
    object->size = size;
    object->data_is_mmapped = data_is_mmapped;

    if (check_range(size, 0, sizeof(struct mach_header_64), "mach header") != 0) {
        return 1;
    }

    const struct mach_header_64 *header = (const struct mach_header_64 *)data;
    if (header->magic != MH_MAGIC_64) {
        return fail("only little-endian 64-bit Mach-O objects are supported");
    }
    if (header->filetype != MH_OBJECT) {
        return fail("input is not a Mach-O object file");
    }

    object->cputype = header->cputype;
    switch (header->cputype) {
        case CPU_TYPE_ARM64:
            object->reloc_unsigned_type = ARM64_RELOC_UNSIGNED;
            object->reloc_subtractor_type = ARM64_RELOC_SUBTRACTOR;
            object->arch_name = "arm64";
            break;
        case CPU_TYPE_X86_64:
            object->reloc_unsigned_type = X86_64_RELOC_UNSIGNED;
            object->reloc_subtractor_type = X86_64_RELOC_SUBTRACTOR;
            object->arch_name = "x86_64";
            break;
        default: {
            char message[96];
            snprintf(message, sizeof(message), "unsupported Mach-O CPU type 0x%x", header->cputype);
            return fail(message);
        }
    }

    size_t load_offset = sizeof(struct mach_header_64);
    bool saw_symtab = false;
    bool saw_dysymtab = false;

    for (uint32_t i = 0; i < header->ncmds; ++i) {
        if (check_range(size, load_offset, sizeof(struct load_command), "load command") != 0) {
            return 1;
        }

        const struct load_command *load_command = (const struct load_command *)(data + load_offset);
        if (load_command->cmdsize < sizeof(struct load_command)) {
            return fail("load command has an invalid size");
        }
        if (check_range(size, load_offset, load_command->cmdsize, "load command payload") != 0) {
            return 1;
        }

        if (load_command->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *segment = (const struct segment_command_64 *)load_command;
            size_t expected_size = sizeof(struct segment_command_64) + (size_t)segment->nsects * sizeof(struct section_64);
            if (load_command->cmdsize < expected_size) {
                return fail("segment command is truncated");
            }

            const struct section_64 *sections = (const struct section_64 *)(segment + 1);
            for (uint32_t section_index = 0; section_index < segment->nsects; ++section_index) {
                if (reserve_bytes((void **)&object->sections, sizeof(SectionInfo), &object->section_capacity, object->section_count + 1) != 0) {
                    return fail("out of memory");
                }

                SectionInfo *section = &object->sections[object->section_count];
                memset(section, 0, sizeof(*section));
                section->index = (uint32_t)object->section_count + 1;
                memcpy(section->name, sections[section_index].sectname, 16);
                section->name[16] = '\0';
                section->addr = sections[section_index].addr;
                section->size = sections[section_index].size;
                section->offset = sections[section_index].offset;
                section->reloff = sections[section_index].reloff;
                section->nreloc = sections[section_index].nreloc;
                object->section_count++;
            }
        } else if (load_command->cmd == LC_SYMTAB) {
            object->symtab_cmd_offset = (uint32_t)load_offset;
            memcpy(&object->symtab, load_command, sizeof(object->symtab));
            saw_symtab = true;
        } else if (load_command->cmd == LC_DYSYMTAB) {
            object->dysymtab_cmd_offset = (uint32_t)load_offset;
            memcpy(&object->dysymtab, load_command, sizeof(object->dysymtab));
            saw_dysymtab = true;
        }

        load_offset += load_command->cmdsize;
    }

    if (!saw_symtab || !saw_dysymtab) {
        return fail("input is missing LC_SYMTAB or LC_DYSYMTAB");
    }

    if (check_range(size, object->symtab.symoff, (size_t)object->symtab.nsyms * sizeof(struct nlist_64), "symbol table") != 0) {
        return 1;
    }
    if (check_range(size, object->symtab.stroff, object->symtab.strsize, "string table") != 0) {
        return 1;
    }

    object->symbols = calloc(object->symtab.nsyms, sizeof(SymbolInfo));
    object->local_symbols_by_section = calloc(object->section_count + 1, sizeof(UInt32Array));
    if (object->symbols == NULL || object->local_symbols_by_section == NULL) {
        return fail("out of memory");
    }
    object->symbol_count = object->symtab.nsyms;

    for (uint32_t i = 0; i < object->symtab.nsyms; ++i) {
        const struct nlist_64 *symbol = (const struct nlist_64 *)(data + object->symtab.symoff + (size_t)i * sizeof(struct nlist_64));
        object->symbols[i].index = i;
        object->symbols[i].n_type = symbol->n_type;
        object->symbols[i].n_sect = symbol->n_sect;
        object->symbols[i].n_value = symbol->n_value;

        if (symbol_is_local_section(&object->symbols[i])) {
            if (object->symbols[i].n_sect > object->section_count) {
                return fail("symbol references a section outside the file");
            }
            if (append_uint32(&object->local_symbols_by_section[object->symbols[i].n_sect], i) != 0) {
                return fail("out of memory");
            }
        }
    }

    return 0;
}

/*
 * Search existing local symbols in the same section for one whose address
 * falls within the candidate's [low_addr, high_addr] interval.  If found,
 * return 1 and set *best_index_out to the closest symbol (by distance from
 * ideal_addr).  Return 0 if no suitable symbol exists.
 */
static int find_best_existing_symbol(const GroupContext *context, const Candidate *candidate, uint32_t *best_index_out) {
    const UInt32Array *locals = &context->local_symbols[context->section_index];
    bool found = false;
    uint64_t best_distance = 0;
    uint32_t best_index = 0;

    for (size_t i = 0; i < locals->count; ++i) {
        uint32_t symbol_index = locals->values[i];
        const SymbolInfo *symbol = &context->symbols[symbol_index];
        int64_t symbol_addr = (int64_t)symbol->n_value;
        if (symbol_addr < candidate->low_addr || symbol_addr > candidate->high_addr) {
            continue;
        }

        uint64_t distance = (uint64_t)llabs(symbol_addr - candidate->ideal_addr);
        if (!found || distance < best_distance) {
            found = true;
            best_distance = distance;
            best_index = symbol_index;
        }
    }

    if (!found) {
        return 0;
    }

    *best_index_out = best_index;
    return 1;
}

/*
 * Find the rightmost safe address inside [low_addr, high_addr].
 *
 * Safe addresses are relocation boundaries gathered from the candidates in
 * the current group, so placing a synthetic symbol there does not create a
 * new atom boundary in the middle of a multi-byte relocated field.
 */
static int choose_new_label_addr(int64_t low_addr, int64_t high_addr, const UInt64Array *safe_addrs, uint64_t *addr_out) {
    size_t lo = 0;
    size_t hi = safe_addrs->count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if ((int64_t)safe_addrs->values[mid] <= high_addr) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo == 0) {
        return fail("failed to find a covering synthetic label");
    }

    uint64_t best_addr = safe_addrs->values[lo - 1];
    if ((int64_t)best_addr < low_addr) {
        return fail("failed to find a covering synthetic label");
    }

    *addr_out = best_addr;
    return 0;
}

/*
 * Sort key for grouping candidates that share the same (section, original
 * subtractor symbol, relocation length).  Within a group the rewriter can
 * share synthetic labels because they need compatible relocation widths.
 */
static int compare_candidate_group(const void *lhs, const void *rhs) {
    const Candidate *a = lhs;
    const Candidate *b = rhs;
    if (a->section_index != b->section_index) {
        return a->section_index < b->section_index ? -1 : 1;
    }
    if (a->subtractor_symbol != b->subtractor_symbol) {
        return a->subtractor_symbol < b->subtractor_symbol ? -1 : 1;
    }
    if (a->length != b->length) {
        return a->length < b->length ? -1 : 1;
    }
    if (a->reloc_addr != b->reloc_addr) {
        return a->reloc_addr < b->reloc_addr ? -1 : 1;
    }
    return 0;
}

static bool same_candidate_group(const Candidate *lhs, const Candidate *rhs) {
    return lhs->section_index == rhs->section_index &&
           lhs->subtractor_symbol == rhs->subtractor_symbol &&
           lhs->length == rhs->length;
}

/*
 * Sort uncovered candidates by high_addr (ascending) for the greedy
 * interval-covering pass.
 */
static int compare_candidate_high_addr(const void *lhs, const void *rhs) {
    const Candidate *const *a = lhs;
    const Candidate *const *b = rhs;
    if ((*a)->high_addr != (*b)->high_addr) {
        return (*a)->high_addr < (*b)->high_addr ? -1 : 1;
    }
    if ((*a)->low_addr != (*b)->low_addr) {
        return (*a)->low_addr < (*b)->low_addr ? -1 : 1;
    }
    return 0;
}

static int compare_label_key(const void *lhs, const void *rhs) {
    const LabelKey *a = lhs;
    const LabelKey *b = rhs;
    if (a->section_index != b->section_index) {
        return a->section_index < b->section_index ? -1 : 1;
    }
    if (a->addr != b->addr) {
        return a->addr < b->addr ? -1 : 1;
    }
    return 0;
}

/*
 * plan_rewrite — Scan all relocation pairs and build a plan for rewriting.
 *
 * Algorithm:
 * 1. Walk every section's relocation table looking for SUBTRACTOR+UNSIGNED
 *    pairs where:
 *      - both entries are extern and non-pcrel
 *      - the subtractor symbol is a local section symbol
 *      - the inline addend exceeds ±max_addend
 *
 * 2. Group candidates by (section, subtractor_symbol, length) so that
 *    candidates in the same group can potentially share a replacement label.
 *
 * 3. For each group, first try to assign each candidate to an *existing*
 *    local symbol whose address falls within the candidate's acceptable
 *    interval.  This avoids creating new symbols when one is already nearby.
 *
 * 4. For remaining ("uncovered") candidates, run a greedy interval-covering
 *    algorithm over *safe* candidate points.  We use relocation-site
 *    addresses from the candidates in the group as possible label positions,
 *    sort by high_addr, and place each new label at the rightmost safe point
 *    inside the current uncovered interval.
 *
 * 5. For each uncovered candidate, find the closest newly placed label and
 *    record the assignment.
 */
static int plan_rewrite(const MachOObject *object, int64_t max_addend, RewritePlan *plan) {
    memset(plan, 0, sizeof(*plan));
    CandidateArray candidates = {0};

    for (size_t section_index = 0; section_index < object->section_count; ++section_index) {
        const SectionInfo *section = &object->sections[section_index];
        /* Skip sections with no relocations or an odd number (pairs expected). */
        if (section->nreloc == 0 || (section->nreloc % 2) != 0) {
            continue;
        }
        if (check_range(object->size, section->reloff, (size_t)section->nreloc * sizeof(struct relocation_info), "relocation table") != 0) {
            free_candidate_array(&candidates);
            return 1;
        }

        /* Iterate relocation pairs (SUBTRACTOR is first, UNSIGNED is second). */
        for (uint32_t pair_index = 0; pair_index < section->nreloc; pair_index += 2) {
            uint32_t first_offset = section->reloff + pair_index * sizeof(struct relocation_info);
            uint32_t second_offset = first_offset + sizeof(struct relocation_info);
            uint32_t first_addr;
            uint32_t first_raw;
            uint32_t second_addr;
            uint32_t second_raw;
            memcpy(&first_addr, object->data + first_offset, sizeof(first_addr));
            memcpy(&first_raw, object->data + first_offset + sizeof(first_addr), sizeof(first_raw));
            memcpy(&second_addr, object->data + second_offset, sizeof(second_addr));
            memcpy(&second_raw, object->data + second_offset + sizeof(second_addr), sizeof(second_raw));

            /* Both entries in a pair must refer to the same r_address. */
            if (first_addr != second_addr) {
                continue;
            }

            RelocFields subtractor;
            RelocFields unsigned_reloc;
            decode_reloc_fields(first_raw, &subtractor);
            decode_reloc_fields(second_raw, &unsigned_reloc);

            /* Only process extern, non-pcrel SUBTRACTOR+UNSIGNED pairs with matching lengths. */
            if (!subtractor.is_extern || !unsigned_reloc.is_extern) {
                continue;
            }
            if (subtractor.pcrel || unsigned_reloc.pcrel) {
                continue;
            }
            if (subtractor.length != unsigned_reloc.length) {
                continue;
            }
            if (subtractor.reloc_type != object->reloc_subtractor_type ||
                unsigned_reloc.reloc_type != object->reloc_unsigned_type) {
                continue;
            }
            if (subtractor.symbolnum >= object->symbol_count || unsigned_reloc.symbolnum >= object->symbol_count) {
                continue;
            }

            /* We only rewrite when the subtractor refers to a local section symbol
             * in the same section as the relocation.  This is the typical pattern
             * for `.long _bar - .` expressions. */
            const SymbolInfo *sub_symbol = &object->symbols[subtractor.symbolnum];
            if (!symbol_is_local_section(sub_symbol) || sub_symbol->n_sect != section->index) {
                continue;
            }

            int64_t addend = 0;
            if (read_signed_addend(object, section, first_addr, subtractor.length, &addend) != 0) {
                free_candidate_array(&candidates);
                return 1;
            }
            /* Addend is within safe range — nothing to do for this pair. */
            if (addend >= -max_addend && addend <= max_addend) {
                continue;
            }

            /*
             * Compute the ideal replacement symbol address and the interval
             * of acceptable addresses.  The addend encodes:
             *   addend = old_symbol_addr - reloc_site_addr
             * so a replacement symbol at ideal_addr would yield addend == 0.
             * We clamp the interval to the section bounds.
             */
            int64_t ideal_addr = (int64_t)sub_symbol->n_value - addend;
            int64_t low_addr = ideal_addr - max_addend;
            int64_t high_addr = ideal_addr + max_addend;
            if (low_addr < (int64_t)section->addr) {
                low_addr = (int64_t)section->addr;
            }
            if (high_addr > section_end_addr(section)) {
                high_addr = section_end_addr(section);
            }
            if (low_addr > high_addr) {
                free_candidate_array(&candidates);
                fprintf(
                    stderr,
                    "error: cannot place a local label in section %s for relocation at 0x%llx\n",
                    section->name,
                    (unsigned long long)first_addr
                );
                return 1;
            }

             Candidate candidate = {
                 .section_index = section->index,
                 .reloc_addr = first_addr,
                 .site_addr = section->addr + first_addr,
                 .reloc_entry_index = pair_index,
                 .subtractor_symbol = subtractor.symbolnum,
                 .unsigned_symbol = unsigned_reloc.symbolnum,
                .length = subtractor.length,
                .old_symbol_addr = (int64_t)sub_symbol->n_value,
                .addend = addend,
                .ideal_addr = ideal_addr,
                .low_addr = low_addr,
                .high_addr = high_addr,
            };
            if (append_candidate(&candidates, &candidate) != 0) {
                free_candidate_array(&candidates);
                return fail("out of memory");
            }
        }
    }

    if (candidates.count == 0) {
        return 0;
    }

    /* Sort candidates into groups of (section, subtractor_symbol, length). */
    qsort(candidates.values, candidates.count, sizeof(Candidate), compare_candidate_group);

    /* Process each group independently. */
    size_t group_start = 0;
    while (group_start < candidates.count) {
        size_t group_end = group_start + 1;
        while (group_end < candidates.count && same_candidate_group(&candidates.values[group_start], &candidates.values[group_end])) {
            group_end++;
        }

        const Candidate *group_head = &candidates.values[group_start];
        GroupContext context = {
            .symbols = object->symbols,
            .local_symbols = object->local_symbols_by_section,
            .section_index = group_head->section_index,
        };

        /* Phase 1: try to assign each candidate to an existing local symbol. */
        Candidate **uncovered = calloc(group_end - group_start, sizeof(Candidate *));
        if (uncovered == NULL) {
            free_candidate_array(&candidates);
            return fail("out of memory");
        }
        size_t uncovered_count = 0;

        for (size_t i = group_start; i < group_end; ++i) {
            uint32_t best_existing = 0;
            if (find_best_existing_symbol(&context, &candidates.values[i], &best_existing)) {
                Assignment assignment = {
                    .candidate = candidates.values[i],
                    .uses_existing_symbol = true,
                    .symbol_index = best_existing,
                };
                if (append_assignment(&plan->assignments, &assignment) != 0) {
                    free(uncovered);
                    free_candidate_array(&candidates);
                    return fail("out of memory");
                }
            } else {
                uncovered[uncovered_count++] = &candidates.values[i];
            }
        }

        /* Phase 2: greedy interval covering for uncovered candidates. */
        if (uncovered_count != 0) {
            /* Sort by high_addr so that a rightmost safe point chosen inside
             * each uncovered interval covers as many subsequent intervals as
             * possible. */
            qsort(uncovered, uncovered_count, sizeof(Candidate *), compare_candidate_high_addr);

            /* Use candidate relocation sites as safe atom boundaries for new
             * labels.  Sorting + deduplicating makes later binary searches
             * cheap even for very large groups. */
            UInt64Array safe_addrs = {0};
            for (size_t i = 0; i < uncovered_count; ++i) {
                if (append_uint64(&safe_addrs, uncovered[i]->site_addr) != 0) {
                    free_uint64_array(&safe_addrs);
                    free(uncovered);
                    free_candidate_array(&candidates);
                    return fail("out of memory");
                }
            }
            qsort(safe_addrs.values, safe_addrs.count, sizeof(uint64_t), compare_uint64);
            deduplicate_uint64_array(&safe_addrs);

            /* Place labels: skip if the last placed label already covers this
             * candidate; otherwise place a new label at the rightmost safe
             * point inside the candidate's interval. */
            UInt64Array new_addrs = {0};
            for (size_t i = 0; i < uncovered_count; ++i) {
                if (new_addrs.count != 0) {
                    uint64_t last = new_addrs.values[new_addrs.count - 1];
                    if ((int64_t)last >= uncovered[i]->low_addr && (int64_t)last <= uncovered[i]->high_addr) {
                        continue;
                    }
                }
                uint64_t chosen_addr = 0;
                if (choose_new_label_addr(uncovered[i]->low_addr, uncovered[i]->high_addr, &safe_addrs, &chosen_addr) != 0) {
                    free_uint64_array(&safe_addrs);
                    free_uint64_array(&new_addrs);
                    free(uncovered);
                    free_candidate_array(&candidates);
                    return 1;
                }
                if (append_uint64(&new_addrs, chosen_addr) != 0) {
                    free_uint64_array(&safe_addrs);
                    free_uint64_array(&new_addrs);
                    free(uncovered);
                    free_candidate_array(&candidates);
                    return fail("out of memory");
                }
            }

            /* Register the newly chosen addresses as synthetic labels. */
            for (size_t i = 0; i < new_addrs.count; ++i) {
                if (append_label_unique(&plan->new_labels, group_head->section_index, new_addrs.values[i]) != 0) {
                    free_uint64_array(&safe_addrs);
                    free_uint64_array(&new_addrs);
                    free(uncovered);
                    free_candidate_array(&candidates);
                    return fail("out of memory");
                }
            }

            /* Phase 3: assign each uncovered candidate to a covering label. */
            for (size_t i = 0; i < uncovered_count; ++i) {
                uint64_t chosen_addr = 0;
                if (choose_new_label_addr(uncovered[i]->low_addr, uncovered[i]->high_addr, &new_addrs, &chosen_addr) != 0) {
                    free_uint64_array(&safe_addrs);
                    free_uint64_array(&new_addrs);
                    free(uncovered);
                    free_candidate_array(&candidates);
                    return 1;
                }

                Assignment assignment = {
                    .candidate = *uncovered[i],
                    .uses_existing_symbol = false,
                    .label_key = {.section_index = group_head->section_index, .addr = chosen_addr},
                };
                if (append_assignment(&plan->assignments, &assignment) != 0) {
                    free_uint64_array(&safe_addrs);
                    free_uint64_array(&new_addrs);
                    free(uncovered);
                    free_candidate_array(&candidates);
                    return fail("out of memory");
                }
            }

            free_uint64_array(&safe_addrs);
            free_uint64_array(&new_addrs);
        }

        free(uncovered);
        group_start = group_end;
    }

    qsort(plan->new_labels.values, plan->new_labels.count, sizeof(LabelKey), compare_label_key);
    plan->rewritten_relocations = plan->assignments.count;
    free_candidate_array(&candidates);
    return 0;
}

/* Look up the output symbol index for a synthetic label by its LabelKey. */
static int find_new_label_index(const RewritePlan *plan, uint32_t insert_at, const LabelKey *key, uint32_t *index_out) {
    for (size_t i = 0; i < plan->new_labels.count; ++i) {
        if (plan->new_labels.values[i].section_index == key->section_index && plan->new_labels.values[i].addr == key->addr) {
            *index_out = insert_at + (uint32_t)i;
            return 0;
        }
    }
    return fail("internal error: missing synthetic label");
}

/*
 * After inserting `added_symbols` new locals at position `insert_at` in the
 * symbol table, every existing symbol at index >= insert_at shifts up.
 */
static uint32_t remap_symbol_index(uint32_t symbol_index, uint32_t insert_at, uint32_t added_symbols) {
    return symbol_index >= insert_at ? symbol_index + added_symbols : symbol_index;
}

/*
 * apply_rewrite — Materialise the plan into a new Mach-O file.
 *
 * Steps:
 * 1. Build the new string table entries (laddend$NNNNNN names) and the
 *    corresponding nlist_64 records for the synthetic symbols.
 *
 * 2. Remap every relocation's symbolnum: any index >= insert_at (the
 *    boundary between locals and extdefs in LC_DYSYMTAB) is shifted up
 *    by the number of added symbols.
 *
 * 3. For each assignment, patch the subtractor relocation to point to the
 *    new (or reused) symbol and recompute the inline addend:
 *        new_addend = old_addend + (new_symbol_addr - old_symbol_addr)
 *
 * 4. Assemble the output file:
 *      [header + load commands + section data]  (unchanged)
 *      [local symbols before insert_at]
 *      [new synthetic nlist_64 entries]
 *      [remaining symbols after insert_at]
 *      [merged string table]
 *      [trailing data after the old string table]
 *
 * 5. Patch LC_SYMTAB (nsyms, stroff, strsize) and LC_DYSYMTAB
 *    (nlocalsym, iextdefsym, iundefsym) in the output header.
 */
static int apply_rewrite(MachOObject *object, const RewritePlan *plan, int64_t max_addend, uint8_t **output_out, size_t *output_size_out, size_t *added_symbols_out) {
    if (plan->assignments.count == 0) {
        uint8_t *copy = malloc(object->size == 0 ? 1 : object->size);
        if (copy == NULL) {
            return fail("out of memory");
        }
        memcpy(copy, object->data, object->size);
        *output_out = copy;
        *output_size_out = object->size;
        *added_symbols_out = 0;
        return 0;
    }

    /*
     * New symbols are inserted at iextdefsym — the boundary between local
     * and external-defined symbols.  This keeps the local symbol range
     * contiguous and only requires bumping iextdefsym/iundefsym indices.
     */
    uint32_t insert_at = object->dysymtab.iextdefsym;
    uint32_t added_symbols = (uint32_t)plan->new_labels.count;

    /* Build the new string table: original strings + synthetic names. */
    size_t old_string_offset = object->symtab.stroff;
    size_t old_string_size = object->symtab.strsize;
    size_t new_string_size = old_string_size;
    for (size_t i = 0; i < plan->new_labels.count; ++i) {
        char name[32];
        int written = snprintf(name, sizeof(name), "laddend$%06zu", i);
        if (written < 0 || (size_t)written + 1 > sizeof(name)) {
            return fail("failed to build synthetic symbol name");
        }
        new_string_size += (size_t)written + 1;
    }

    uint8_t *new_strings = malloc(new_string_size);
    if (new_strings == NULL) {
        return fail("out of memory");
    }
    memcpy(new_strings, object->data + old_string_offset, old_string_size);

    /* Build nlist_64 entries for the synthetic symbols.  Each is N_SECT
     * (local, section-defined) with n_value set to the label's address. */
    struct nlist_64 *new_symbol_entries = calloc(plan->new_labels.count, sizeof(struct nlist_64));
    if (new_symbol_entries == NULL) {
        free(new_strings);
        return fail("out of memory");
    }

    size_t string_cursor = old_string_size;
    for (size_t i = 0; i < plan->new_labels.count; ++i) {
        char name[32];
        int written = snprintf(name, sizeof(name), "laddend$%06zu", i);
        if (written < 0 || (size_t)written + 1 > sizeof(name)) {
            free(new_symbol_entries);
            free(new_strings);
            return fail("failed to build synthetic symbol name");
        }
        memcpy(new_strings + string_cursor, name, (size_t)written + 1);
        new_symbol_entries[i].n_un.n_strx = (uint32_t)string_cursor;
        new_symbol_entries[i].n_type = N_SECT;
        new_symbol_entries[i].n_sect = (uint8_t)plan->new_labels.values[i].section_index;
        new_symbol_entries[i].n_desc = 0;
        new_symbol_entries[i].n_value = plan->new_labels.values[i].addr;
        string_cursor += (size_t)written + 1;
    }

    /*
     * Pass 1: remap all existing relocation symbolnums to account for the
     * inserted symbols shifting indices at and above insert_at.
     */
    for (size_t section_index = 0; section_index < object->section_count; ++section_index) {
        const SectionInfo *section = &object->sections[section_index];
        if (section->nreloc == 0) {
            continue;
        }
        for (uint32_t reloc_index = 0; reloc_index < section->nreloc; ++reloc_index) {
            uint32_t reloc_offset = section->reloff + reloc_index * sizeof(struct relocation_info);
            uint32_t raw_word;
            memcpy(&raw_word, object->data + reloc_offset + sizeof(uint32_t), sizeof(raw_word));
            RelocFields fields;
            decode_reloc_fields(raw_word, &fields);
            if (fields.is_extern && fields.symbolnum >= insert_at) {
                uint32_t remapped = encode_reloc_symbol(raw_word, remap_symbol_index(fields.symbolnum, insert_at, added_symbols));
                memcpy(object->data + reloc_offset + sizeof(uint32_t), &remapped, sizeof(remapped));
            }
        }
    }

    /*
     * Pass 2: for each assignment, patch the subtractor symbol and addend.
     * The new addend is:  old_addend + (new_symbol_addr - old_symbol_addr)
     * because:  value = unsigned_sym + addend - subtractor_sym
     * and we're replacing subtractor_sym, so the addend must compensate.
     */
    for (size_t i = 0; i < plan->assignments.count; ++i) {
        const Assignment *assignment = &plan->assignments.values[i];
        const Candidate *candidate = &assignment->candidate;
        SectionInfo *section = &object->sections[candidate->section_index - 1];

        uint32_t new_symbol_index = 0;
        int64_t new_symbol_addr = 0;
        if (assignment->uses_existing_symbol) {
            new_symbol_index = remap_symbol_index(assignment->symbol_index, insert_at, added_symbols);
            new_symbol_addr = (int64_t)object->symbols[assignment->symbol_index].n_value;
        } else {
            if (find_new_label_index(plan, insert_at, &assignment->label_key, &new_symbol_index) != 0) {
                free(new_symbol_entries);
                free(new_strings);
                return 1;
            }
            new_symbol_addr = (int64_t)assignment->label_key.addr;
        }

        int64_t new_addend = candidate->addend + (new_symbol_addr - candidate->old_symbol_addr);
        if (new_addend < -max_addend || new_addend > max_addend) {
            free(new_symbol_entries);
            free(new_strings);
            return fail("rewritten addend still exceeds the configured limit");
        }
        if (write_signed_addend(object, section, candidate->reloc_addr, candidate->length, new_addend) != 0) {
            free(new_symbol_entries);
            free(new_strings);
            return 1;
        }

        uint32_t subtractor_offset = section->reloff + candidate->reloc_entry_index * sizeof(struct relocation_info);
        uint32_t unsigned_offset = subtractor_offset + sizeof(struct relocation_info);
        uint32_t subtractor_raw;
        uint32_t unsigned_raw;
        memcpy(&subtractor_raw, object->data + subtractor_offset + sizeof(uint32_t), sizeof(subtractor_raw));
        memcpy(&unsigned_raw, object->data + unsigned_offset + sizeof(uint32_t), sizeof(unsigned_raw));

        subtractor_raw = encode_reloc_symbol(subtractor_raw, new_symbol_index);
        unsigned_raw = encode_reloc_symbol(unsigned_raw, remap_symbol_index(candidate->unsigned_symbol, insert_at, added_symbols));
        memcpy(object->data + subtractor_offset + sizeof(uint32_t), &subtractor_raw, sizeof(subtractor_raw));
        memcpy(object->data + unsigned_offset + sizeof(uint32_t), &unsigned_raw, sizeof(unsigned_raw));
    }

    /*
     * Assemble the output buffer:
     *   1. Everything before the symbol table (headers, section data, relocs)
     *   2. Symbol table with new entries spliced in at insert_at
     *   3. Merged string table (old + new names)
     *   4. Any trailing data after the original string table
     */
    size_t old_symbol_blob_size = (size_t)object->symtab.nsyms * sizeof(struct nlist_64);
    size_t new_symbol_blob_size = ((size_t)object->symtab.nsyms + added_symbols) * sizeof(struct nlist_64);
    size_t suffix_offset = old_string_offset + old_string_size;
    size_t suffix_size = object->size - suffix_offset;
    size_t output_size = object->symtab.symoff + new_symbol_blob_size + new_string_size + suffix_size;
    uint8_t *output = malloc(output_size == 0 ? 1 : output_size);
    if (output == NULL) {
        free(new_symbol_entries);
        free(new_strings);
        return fail("out of memory");
    }

    memcpy(output, object->data, object->symtab.symoff);
    size_t out_offset = object->symtab.symoff;
    size_t local_prefix_size = (size_t)insert_at * sizeof(struct nlist_64);
    size_t ext_suffix_size = old_symbol_blob_size - local_prefix_size;
    memcpy(output + out_offset, object->data + object->symtab.symoff, local_prefix_size);
    out_offset += local_prefix_size;
    memcpy(output + out_offset, new_symbol_entries, (size_t)added_symbols * sizeof(struct nlist_64));
    out_offset += (size_t)added_symbols * sizeof(struct nlist_64);
    memcpy(output + out_offset, object->data + object->symtab.symoff + local_prefix_size, ext_suffix_size);
    out_offset += ext_suffix_size;
    memcpy(output + out_offset, new_strings, new_string_size);
    out_offset += new_string_size;
    memcpy(output + out_offset, object->data + suffix_offset, suffix_size);

    /* Patch LC_SYMTAB and LC_DYSYMTAB in the output to reflect the new
     * symbol count, string table offset/size, and local/extdef/undef ranges. */
    struct symtab_command *symtab = (struct symtab_command *)(output + object->symtab_cmd_offset);
    symtab->symoff = object->symtab.symoff;
    symtab->nsyms = object->symtab.nsyms + added_symbols;
    symtab->stroff = object->symtab.symoff + (uint32_t)new_symbol_blob_size;
    symtab->strsize = (uint32_t)new_string_size;

    struct dysymtab_command *dysymtab = (struct dysymtab_command *)(output + object->dysymtab_cmd_offset);
    dysymtab->ilocalsym = object->dysymtab.ilocalsym;
    dysymtab->nlocalsym = object->dysymtab.nlocalsym + added_symbols;
    dysymtab->iextdefsym = object->dysymtab.iextdefsym + added_symbols;
    dysymtab->nextdefsym = object->dysymtab.nextdefsym;
    dysymtab->iundefsym = object->dysymtab.iundefsym + added_symbols;
    dysymtab->nundefsym = object->dysymtab.nundefsym;

    free(new_symbol_entries);
    free(new_strings);
    *output_out = output;
    *output_size_out = output_size;
    *added_symbols_out = added_symbols;
    return 0;
}

static int write_all(int fd, const uint8_t *data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        ssize_t written = write(fd, data + offset, size - offset);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        offset += (size_t)written;
    }
    return 0;
}

/*
 * Write output to disk atomically: write to a temporary file in the same
 * directory, then rename over the target.  Preserves permissions from the
 * output path (or input path as fallback).
 */
static int write_output_file(const char *input_path, const char *output_path, const uint8_t *data, size_t size) {
    char directory[PATH_MAX];
    const char *slash = strrchr(output_path, '/');
    if (slash == NULL) {
        strcpy(directory, ".");
    } else {
        size_t length = (size_t)(slash - output_path);
        if (length == 0) {
            strcpy(directory, "/");
        } else {
            if (length >= sizeof(directory)) {
                return fail("output path is too long");
            }
            memcpy(directory, output_path, length);
            directory[length] = '\0';
        }
    }

    char temporary_path[PATH_MAX];
    int template_length = snprintf(temporary_path, sizeof(temporary_path), "%s/.macho_addend_rewriter.XXXXXX", directory);
    if (template_length < 0 || (size_t)template_length >= sizeof(temporary_path)) {
        return fail("temporary file path is too long");
    }

    int fd = mkstemp(temporary_path);
    if (fd < 0) {
        return fail_errno("mkstemp");
    }

    struct stat st;
    if (stat(output_path, &st) == 0) {
        (void)fchmod(fd, st.st_mode & 0777);
    } else if (stat(input_path, &st) == 0) {
        (void)fchmod(fd, st.st_mode & 0777);
    }

    if (write_all(fd, data, size) != 0) {
        int saved_errno = errno;
        close(fd);
        unlink(temporary_path);
        errno = saved_errno;
        return fail_errno("write");
    }
    if (close(fd) != 0) {
        int saved_errno = errno;
        unlink(temporary_path);
        errno = saved_errno;
        return fail_errno("close");
    }
    if (rename(temporary_path, output_path) != 0) {
        int saved_errno = errno;
        unlink(temporary_path);
        errno = saved_errno;
        return fail_errno("rename");
    }
    return 0;
}

static void print_usage(FILE *stream, const char *program_name) {
    fprintf(
        stream,
        "Usage: %s [--max-addend VALUE] [-o OUTPUT] [--verbose] INPUT\n"
        "\n"
        "Rewrite 64-bit Mach-O object relocations so large subtractor addends are\n"
        "split across nearby synthetic local symbols.\n",
        program_name
    );
}

int main(int argc, char **argv) {
    static const struct option long_options[] = {
        {"output", required_argument, NULL, 'o'},
        {"max-addend", required_argument, NULL, 'm'},
        {"verbose", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0},
    };

    const char *output_path = NULL;
    /*
     * Default threshold: 0x3f0000 (4 MiB - 64 KiB).
     *
     * A real-world NativeAOT iOS object still triggered the linker assert at
     * 0x400000 and even at 0x3fc000, but linked successfully at 0x3fa000 and
     * below once synthetic labels were restricted to safe relocation
     * boundaries.  Keep a little extra margin by default.
     */
    int64_t max_addend = 0x3f0000;
    bool verbose = false;

    while (true) {
        int option = getopt_long(argc, argv, "o:m:vh", long_options, NULL);
        if (option == -1) {
            break;
        }

        switch (option) {
            case 'o':
                output_path = optarg;
                break;
            case 'm': {
                bool ok = false;
                max_addend = parse_integer(optarg, "--max-addend", &ok);
                if (!ok) {
                    return 1;
                }
                break;
            }
            case 'v':
                verbose = true;
                break;
            case 'h':
                print_usage(stdout, argv[0]);
                return 0;
            default:
                print_usage(stderr, argv[0]);
                return 1;
        }
    }

    if (optind + 1 != argc) {
        print_usage(stderr, argv[0]);
        return 1;
    }

    const char *input_path = argv[optind];
    if (output_path == NULL) {
        output_path = input_path;
    }

    uint8_t *file_data = NULL;
    size_t file_size = 0;
    if (mmap_file(input_path, &file_data, &file_size) != 0) {
        return 1;
    }

    MachOObject object;
    memset(&object, 0, sizeof(object));
    if (parse_macho_object(file_data, file_size, true, &object) != 0) {
        free_macho_object(&object);
        return 1;
    }

    RewritePlan plan;
    if (plan_rewrite(&object, max_addend, &plan) != 0) {
        free_macho_object(&object);
        return 1;
    }

    uint8_t *output_data = NULL;
    size_t output_size = 0;
    size_t added_symbols = 0;
    if (apply_rewrite(&object, &plan, max_addend, &output_data, &output_size, &added_symbols) != 0) {
        free_rewrite_plan(&plan);
        free_macho_object(&object);
        return 1;
    }

    int exit_code = write_output_file(input_path, output_path, output_data, output_size);
    if (exit_code == 0) {
        if (verbose) {
            printf(
                "rewrote %zu relocation pairs and added %zu local symbols\n",
                plan.rewritten_relocations,
                added_symbols
            );
        } else if (plan.rewritten_relocations != 0) {
            printf(
                "patched %s: rewrote %zu relocation pairs with %zu synthetic local symbols\n",
                input_path,
                plan.rewritten_relocations,
                added_symbols
            );
        } else {
            printf("patched %s: no large relocation addends found\n", input_path);
        }
    }

    free(output_data);
    free_rewrite_plan(&plan);
    free_macho_object(&object);
    return exit_code;
}
