#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]){

	char *line, *col1 = (char *)malloc(100 * sizeof(char)), *col2 = (char *)malloc(100 * sizeof(char));
	char *command = (char *)malloc(100 * sizeof(char));
	int line1 = 0;
	if(argc < 8){
		printf("\nUsage : extractFromCSV inputFile outputFile lookFor count header=true/false inputDir outputDir\n");
		exit(0);
	}

	FILE *inputFilePtr = NULL, *outputFilePtr = NULL;
	if(!(inputFilePtr = fopen(argv[1], "r"))){
		printf("\nFile: %s does not found\n", argv[1]);
		exit(0);
	}
	if(!(outputFilePtr = fopen(argv[2], "a+"))){
		printf("\nCould not open %s\n", argv[2]);
		exit(0);
	}
	if(!strcmp(argv[5], "true")){
		fscanf(inputFilePtr, "%s", line);
		fprintf(outputFilePtr, "%s", line);
		printf("\n%s", line);
		line1 = 1;

	}
	long count = strtol(argv[4], NULL, 10);
	strcpy(command, "mkdir ");
	strcat(command, argv[7]);
	system(command);
	while(fscanf(inputFilePtr, "%s", line) == 1 && count){
		char *op = (char *)malloc(strlen(line) * sizeof(char));
		strcpy(op, line);
		col1 = strtok(line, ",");
		col2 = strtok(NULL, "\n");
		if(!strcmp(col2, argv[3])){
			if(line1)
				fprintf(outputFilePtr, "\n%s", op);
			else{
				fprintf(outputFilePtr, "%s", op);
				line1 = 1;
			}
			strcpy(command, "cp ");
			strcat(command, argv[6]);
			strcat(command, "/");
			strcat(command, col1);
			strcat(command, ".jpeg ");
			strcat(command, argv[7]);
			system(command);
			printf("\n%s", op); 
			count--;
		}
	}
	fclose(inputFilePtr);
	fclose(outputFilePtr);
	return 0;
}
