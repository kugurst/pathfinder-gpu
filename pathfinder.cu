
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <windows.h>

int main(int argc, char** argv)
{
	printf("%s\n", argv);
	Sleep(1000);
	return 0;
}