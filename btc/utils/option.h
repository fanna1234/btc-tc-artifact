
#pragma once

#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>

namespace btc {

struct Config {
    std::string input_file;

    bool verify  = false;
    int  warmup  = 1;
    int  repeat  = 5;
};

std::string option_hints =
    "              [-i input_file]\n"
    "              [-v verify (1 or 0)]\n"
    "              [-w warmup (default 1)]\n"
    "              [-r repeat (default 5)]\n";

auto program_options(int argc, char* argv[])
{
    Config config;
    int    opt;
    if (argc == 1) {
        printf("Usage: %s ... \n%s", argv[0], option_hints.c_str());
        std::exit(EXIT_FAILURE);
    }
    while ((opt = getopt(argc, argv, "v:i:w:r:")) != -1) {
        switch (opt) {
            case 'i':
                config.input_file = optarg;
                break;
            case 'v':
                config.verify = std::stoi(optarg);
                break;
            case 'w':
                config.warmup = std::stoi(optarg);
                break;
            case 'r':
                config.repeat = std::stoi(optarg);
                break;

            default:
                printf("Usage: %s ... \n%s", argv[0], option_hints.c_str());
                exit(EXIT_FAILURE);
        }
    }
    printf("\n-----------------config-----------------\n");
    if (!config.input_file.empty()) {
        printf("input path: %s\n", config.input_file.c_str());
    }
    else {
        printf("input file is not specified\n");
        exit(EXIT_FAILURE);
    }
    if (config.verify == 1) {
        printf("verify with CPU result\n");
    }
    printf("warmup: %d, repeat: %d\n", config.warmup, config.repeat);
    printf("----------------------------------------\n");
    printf("\n");
    return config;
}
}  // namespace btc
