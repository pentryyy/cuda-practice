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
nvc++ -acc -gpu=mem:managed main.cpp -o compile/main
./compile/main
```

Вместо `main` указываем название желаемого файла.  
Или скомпилировать все исполняемые файлы через `compile.sh`.  
Убедитесь, что файл сохранен с Unix-форматом окончания строк (LF), а не Windows (CRLF).

```
sed -i 's/\r$//' compile.sh
./compile.sh
```
