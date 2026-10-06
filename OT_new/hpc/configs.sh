#!/bin/bash
# Named experiment configurations for run_cifar10.py.  Each prints its flags.

config_flags() {
    case "$1" in
        paper_gamma)
            # The setting of the current paper figure (relative-change inner
            # stopping), now also logging Gamma_k.
            echo "--coord-inner-iterations 50 --coord-inner-tol 1e-3 --log-gamma"
            ;;
        certified)
            # Inner solves stop at the certified eps_k rule; budgets sized for
            # about 30k MatVec passes.
            echo "--inner-stopping certificate --log-gamma" \
                 "--uniform-sweeps 300 --gs-sweeps 300" \
                 "--full-outer-iterations 100 --full-inner-iterations 200" \
                 "--coord-inner-iterations 200"
            ;;
        *)
            echo "Unknown configuration: $1" >&2
            return 2
            ;;
    esac
}
