/* Command-line arguments shared by the tools: "--name value" options and "--name" flags.
 * An argument that is neither is an error, never ignored. */
#include "s1.h"

#include <string.h>

static bool listed(const char *const *names, const char *arg)
{
    for (; names && *names; names++) {
        if (strcmp(*names, arg) == 0) {
            return true;
        }
    }
    return false;
}

int s1_args_check(int argc, char **argv, const char *const *options, const char *const *flags)
{
    for (int i = 1; i < argc; i++) {
        if (listed(options, argv[i]) && i + 1 < argc) {
            i++; /* skip the option's value */
        } else if (!listed(flags, argv[i])) {
            fprintf(stderr, "%s: unknown or incomplete argument \"%s\"\n", argv[0], argv[i]);
            return -1;
        }
    }
    return 0;
}

const char *s1_arg_value(int argc, char **argv, const char *name)
{
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return NULL;
}

bool s1_arg_flag(int argc, char **argv, const char *name)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            return true;
        }
    }
    return false;
}
