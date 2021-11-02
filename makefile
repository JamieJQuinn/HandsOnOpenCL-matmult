exe=exe
source_files=host.cpp
CFLAGS=-std=c++17 -pthread -O0 -g
LDFLAGS=-lOpenCL

all: ${exe}

${exe}: ${source_files}
	g++ $< ${CFLAGS} ${LDFLAGS} -o $@

.PHONY: clean
clean:
	${RM} ${exe}
