#python3 --version
#poetry run python3 tests/say_hello.py
#poetry run python3 tests/memory_leak.py
valgrind --leak-check=full --quiet --log-file=data/output_valgrind.txt --show-leak-kinds=all --track-origins=yes python3 tests/memory_leak.py