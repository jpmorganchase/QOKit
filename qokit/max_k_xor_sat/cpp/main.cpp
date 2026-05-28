/**
 * CLI for QAOA symmetric tree contraction.
 *
 * Usage:
 *   ./qaoa_contract --k 2 --D 3 --p 5 --gammas 0.1,0.2,0.3,0.4,0.5 --betas 0.1,0.2,0.3,0.4,0.5
 *   ./qaoa_contract --k 2 --D 3 --p 5 --gammas ... --betas ... --grad
 *   ./qaoa_contract --k 2 --D 3 --p 5 --gammas ... --betas ... --precision dd
 *   echo '{"gammas": [...], "betas": [...]}' | ./qaoa_contract --k 2 --D 3 --p 5 --stdin
 *   ./qaoa_contract --k 2 --D 3 --p 5 --gammas ... --betas ... --verbose
 */

#include "contract.h"
#include "grad.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>


// Minimal JSON output (no external dependency required)
static void print_json_double(const char* key, double val, bool comma = true) {
    printf("\"%s\": %.15g%s", key, val, comma ? ", " : "");
}

static void print_json_array(const char* key, const double* vals, int n, bool comma = true) {
    printf("\"%s\": [", key);
    for (int i = 0; i < n; i++) {
        printf("%.15g%s", vals[i], (i < n - 1) ? ", " : "");
    }
    printf("]%s", comma ? ", " : "");
}


static std::vector<double> parse_csv(const std::string& s) {
    std::vector<double> vals;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (!token.empty()) {
            vals.push_back(std::stod(token));
        }
    }
    return vals;
}


// Minimal JSON parser for stdin mode: extract "gammas": [...] and "betas": [...]
static bool parse_json_angles(const std::string& json,
                              std::vector<double>& gammas,
                              std::vector<double>& betas) {
    auto extract_array = [](const std::string& json, const std::string& key,
                           std::vector<double>& out) -> bool {
        std::string search = "\"" + key + "\"";
        auto pos = json.find(search);
        if (pos == std::string::npos) return false;
        pos = json.find('[', pos);
        if (pos == std::string::npos) return false;
        auto end = json.find(']', pos);
        if (end == std::string::npos) return false;
        std::string arr_str = json.substr(pos + 1, end - pos - 1);
        out = parse_csv(arr_str);
        return true;
    };

    return extract_array(json, "gammas", gammas) &&
           extract_array(json, "betas", betas);
}


static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s --k K --D D --p P --gammas g1,g2,... --betas b1,b2,...\n", prog);
    fprintf(stderr, "       %s --k K --D D --p P --stdin  (read JSON from stdin)\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --k K          Hyperedge size\n");
    fprintf(stderr, "  --D D          Vertex degree\n");
    fprintf(stderr, "  --p P          QAOA depth\n");
    fprintf(stderr, "  --gammas       Comma-separated phase angles\n");
    fprintf(stderr, "  --betas        Comma-separated mixer angles\n");
    fprintf(stderr, "  --grad         Compute gradient\n");
    fprintf(stderr, "  --precision dd Use double-double precision\n");
    fprintf(stderr, "  --stdin        Read angles from stdin as JSON\n");
    fprintf(stderr, "  --verbose      Print timing to stderr\n");
    fprintf(stderr, "  --lightcone    Print light cone size and exit\n");
}


int main(int argc, char** argv) {
    int k = 0, D = 0, p = 0;
    std::string gammas_str, betas_str;
    bool do_grad = false;
    bool use_dd = false;
    bool use_stdin = false;
    bool verbose = false;
    bool lightcone_only = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--D") == 0 && i + 1 < argc) {
            D = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--p") == 0 && i + 1 < argc) {
            p = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gammas") == 0 && i + 1 < argc) {
            gammas_str = argv[++i];
        } else if (strcmp(argv[i], "--betas") == 0 && i + 1 < argc) {
            betas_str = argv[++i];
        } else if (strcmp(argv[i], "--grad") == 0) {
            do_grad = true;
        } else if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
            std::string prec = argv[++i];
            use_dd = (prec == "dd");
        } else if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--lightcone") == 0) {
            lightcone_only = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (k <= 0 || D <= 0 || p < 0) {
        fprintf(stderr, "Error: --k, --D, --p are required and must be positive.\n");
        usage(argv[0]);
        return 1;
    }

    if (lightcone_only) {
        printf("{\"light_cone_size\": %lld}\n", light_cone_size(p, D, k));
        return 0;
    }

    std::vector<double> gammas, betas;

    if (use_stdin) {
        // Read JSON from stdin
        std::string json_input;
        std::string line;
        while (std::getline(std::cin, line)) {
            json_input += line;
        }
        if (!parse_json_angles(json_input, gammas, betas)) {
            fprintf(stderr, "Error: failed to parse JSON from stdin.\n");
            return 1;
        }
    } else {
        if (gammas_str.empty() || betas_str.empty()) {
            fprintf(stderr, "Error: --gammas and --betas are required (or use --stdin).\n");
            usage(argv[0]);
            return 1;
        }
        gammas = parse_csv(gammas_str);
        betas = parse_csv(betas_str);
    }

    if ((int)gammas.size() != p || (int)betas.size() != p) {
        fprintf(stderr, "Error: gammas and betas must each have length p=%d "
                        "(got %zu and %zu).\n", p, gammas.size(), betas.size());
        return 1;
    }

    if (do_grad && !use_dd) {
        std::vector<double> grad_g(p), grad_b(p);
        double val = contract_with_grad(gammas.data(), betas.data(), p, D, k,
                                        grad_g.data(), grad_b.data());
        printf("{");
        print_json_double("expectation", val);
        print_json_double("objective", (1.0 - val) / 2.0);
        print_json_array("grad_gammas", grad_g.data(), p);
        print_json_array("grad_betas", grad_b.data(), p, false);
        printf("}\n");
    } else {
        double val = contract_symmetric_tree(gammas.data(), betas.data(),
                                             p, D, k, use_dd, verbose);
        printf("{");
        print_json_double("expectation", val);
        print_json_double("objective", (1.0 - val) / 2.0, false);
        printf("}\n");
    }

    return 0;
}
