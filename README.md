# Сборка проекта

```
mkdir build
cd build
cmake ..
```

# Сборка (для Visual Studio)

```
cmake --build . --target ALL_BUILD --config Release
```

# Запуск после сборки

Выполняется из директории `build`.

```
./Release/main
```

Вместо `main` указываем название желаемого файла.

# Запуск OpenACC

Выполняется из среды linux с использованием NVIDIA HPC SDK.

```
nvc++ -acc -gpu=managed main.cpp -o main
./main
```

Вместо `main` указываем название желаемого файла.
