#define _CRT_SECURE_NO_WARNINGS
#include "Header.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
#pragma region ONLY_FOR_DEBUG
	argc = 5;
	char* temp[5];
	temp[0] = _strdup("1712842_Lab03.exe");
	temp[1] = _strdup("D://test.jpeg");
	temp[2] = _strdup("--blob");
	temp[3] = _strdup("150");
	temp[4] = _strdup("0.04");
	argv = temp;
#pragma endregion
	try {
		char *command, *inputPath, *cmdArgument1, *cmdArgument2;
		inputPath = argv[1];
		command = argv[2];
		cmdArgument1 = argv[3];
		cmdArgument2 = argv[4];

		Mat input, srcGray;
		Mat output;

		input = imread(inputPath);
		cvtColor(input, srcGray, COLOR_BGR2GRAY);

		//Check input image
		if (!input.data)
		{
			cout << "Error in input image!" << endl;
			return -1;
		}

		//RGB to Gray image
		if (strcmp(command, "--harrist") == 0)
		{
			int thres = atoi(cmdArgument1);
			int k = atoi(cmdArgument2);
			output = detectHarrist(srcGray, thres, k);
		}
		//Gray image to RGB
		else if (strcmp(command, "--blob") == 0)
		{
			int threshold = atoi(cmdArgument1);
			output = detectBlob(srcGray, threshold);
		}
		//Change the brightness
		else if (strcmp(command, "--DoG") == 0)
		{
			int threshold = atoi(cmdArgument1);
			output = detectBlob(srcGray, threshold);
		}
		//Change contrast
		else if (strcmp(command, "--SIFT") == 0)
		{
			int threshold = atoi(cmdArgument1);
			output = detectBlob(srcGray, threshold);
		}

		//Show result
		if (output.data)
		{
			imshow("Source Image", input);
			imshow("Destination Image", output);
			waitKey(0);
		}
		else
		{
			throw "Error:..........";
		}
	}
	catch (const char* msg) {
		cout << msg << endl;
		system("pause");
	}
#pragma region ONLY_FOR_DEBUG
	free(temp[0]);
	free(temp[1]);
	free(temp[2]);
	free(temp[3]);
	free(temp[4]);
#pragma endregion
	return 0;
}