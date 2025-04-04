//
// Created by Edoardo Leali on 26/03/25.
//

#ifndef MPIHELPER_HPP
#define MPIHELPER_HPP

#include <mpi.h>

class MPIHelper {
public:
    static MPIHelper& instance(int& argc, char**& argv) {
        static MPIHelper singleton(argc, argv);
        return singleton;
    }

private:
    bool initializedHere_;

    MPIHelper(int& argc, char**& argv) : initializedHere_(false) {
        int flag;
        MPI_Initialized(&flag);
        if (!flag) {
            MPI_Init(&argc, &argv);
            initializedHere_ = true;
        }
    }

    ~MPIHelper() {
        if (initializedHere_) {
            MPI_Finalize();
        }
    }

    MPIHelper(const MPIHelper&) = delete;
    MPIHelper& operator=(const MPIHelper&) = delete;
};

#endif // MPIHELPER_HPP
