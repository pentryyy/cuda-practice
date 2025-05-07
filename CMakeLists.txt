cmake_minimum_required(VERSION 3.18)
project(cuda-practice LANGUAGES CXX CUDA)

# Включаем поддержку современных стандартов
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

add_executable(main main.cu)

# Для Visual Studio добавляем настройки
if(MSVC)
    target_compile_options(main
        PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4 -Wno-deprecated-gpu-targets>
    )
    target_compile_definitions(main
        PRIVATE 
        _CRT_SECURE_NO_WARNINGS
    )
endif()