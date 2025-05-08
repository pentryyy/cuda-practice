cmake_minimum_required(VERSION 3.18)
project(cuda-practice LANGUAGES CXX CUDA)

# Включаем поддержку C++17
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Указываем архитектуры CUDA
set(CMAKE_CUDA_ARCHITECTURES 
    50 52 53    # Maxwell
    60 61 62    # Pascal
    70 72       # Volta
    75          # Turing
    80 86 87    # Ampere
    89          # Ada Lovelace
    90          # Hopper
)

# Исходники для 1-ой лекции
add_executable(adding_vectors lecture/1/adding_vectors.cu)

# Исходники для 2-ой лекции
add_executable(complex_function lecture/2/complex_function.cu)
add_executable(cuda_streams     lecture/2/cuda_streams.cu)
add_executable(pinned_memory    lecture/2/pinned_memory.cu)

# Исходники для 3-ей лекции
add_executable(solver_slau  lecture/3/solver_slau.cu)
add_executable(solver_chaos lecture/3/solver_chaos.cu)

# Для Visual Studio добавляем настройки
if(MSVC)
    # Список всех целей
    set(TARGETS 
        adding_vectors 
        complex_function cuda_streams pinned_memory
        solver_slau solver_chaos
    )

    # Применяем настройки к каждому элементу списка
    foreach(target IN LISTS TARGETS)
        target_compile_options(${target}
            PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4 -Wno-deprecated-gpu-targets>
        )
        target_compile_definitions(${target}
            PRIVATE 
            _CRT_SECURE_NO_WARNINGS
        )
    endforeach()
endif()