#include <stdlib.h>

long __isoc23_strtol(const char *nptr, char **endptr, int base);
unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base);

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
    return strtol(nptr, endptr, base);
}
unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base) {
    return strtoul(nptr, endptr, base);
}
