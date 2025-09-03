#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "toml.h"
#include "utils.h"

int main(void) {
    FILE* fp;
    char errbuf[200];
    toml_table_t* conf;

    /* Open file */
    char *path = relative_path_read("../../apps/config.toml");
    if (!path) {
        return 1;
    }
    fp = fopen(path, "r");
    if (!fp) {
        perror("fopen");
        return 1;
    }

    /* Parse */
    conf = toml_parse_file(fp, errbuf, sizeof(errbuf));
    fclose(fp);

    if (!conf) {
        fprintf(stderr, "Error: %s\n", errbuf);
        return 1;
    }

    /* Access server.host */
    toml_table_t* server = toml_table_in(conf, "server");
    if (server) {
        char* host;
        if (toml_rtos(toml_raw_in(server, "host"), &host) == 0) {
            printf("Server host: %s\n", host);
            free((void*)host);
        }
        int64_t port;
        if (toml_rtoi(toml_raw_in(server, "port"), &port) == 0) {
            printf("Server port: %" PRId64 "\n", port);
        }
    }

    /* Access database.user */
    toml_table_t* db = toml_table_in(conf, "database");
    if (db) {
        char* user;
        if (toml_rtos(toml_raw_in(db, "user"), &user) == 0) {
            printf("DB user: %s\n", user);
            free((void*)user);
        }
    }

    toml_free(conf);
    return 0;
}