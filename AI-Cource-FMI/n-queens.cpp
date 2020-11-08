#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>  
#include <random>
#include <chrono>
#include <list>
//#include <unordered_map>

int sizeOfBoardd;
int* queens;
int* rows;
int* diagonal1;
int* diagonal2;
int* container;

const int k = 1000;
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

inline int getDiag1Index(const int x, const int y) {
	return x - y + sizeOfBoardd - 1;
}

inline int getDiag2Index(const int x, const int y) {
	return x + y;
}

inline int getConflicts(int x, int y) {
	return diagonal1[getDiag1Index(x, y)] +
		diagonal2[getDiag2Index(x, y)] +
		rows[y];
}

inline void updateConflictStatistics(const int queen, const int newRow) {
	//if the queen wasnt place on the board before
	if (queens[queen] != -1) {
		rows[queens[queen]]--;
		diagonal1[getDiag1Index(queen, queens[queen])]--;
		diagonal2[getDiag2Index(queen, queens[queen])]--;
	}

	queens[queen] = newRow;

	rows[newRow]++;
	diagonal1[getDiag1Index(queen, newRow)]++;
	diagonal2[getDiag2Index(queen, newRow)]++;
}


void initializeBoard() {

	//initialize board statistics
	for (int i = 0; i < 2 * sizeOfBoardd - 1; i++) {
		diagonal1[i] = 0;
		diagonal2[i] = 0;
	}

	for (int i = 0; i < sizeOfBoardd; i++) {
		rows[i] = 0;
		queens[i] = -1;
	}
	
	//populate board with the queens
	updateConflictStatistics(0, rng() % sizeOfBoardd);

	for (int i = 1; i < sizeOfBoardd; i++) {
		int frontCollisions = INT32_MAX;
		int bestCount = 0;
		for (int k = 0; k < sizeOfBoardd; k++) {
			int collisions = getConflicts(i, k);

			if (bestCount == 0 || frontCollisions == collisions) {
				container[bestCount] = k;
				frontCollisions = collisions;
				bestCount++;
			}
			else if (frontCollisions > collisions) {
				frontCollisions = collisions;
				container[0] = k;
				bestCount = 1;
			}
		}

		//pick random from best possible scenarios
		int winner = rng() % bestCount;
		updateConflictStatistics(i, container[winner]);

		bestCount = 0;
	}
}


inline std::pair<int,int> getMaxConflictQueen() {
	int frontConflicts = 0;
	int bestCount = 0;
	for (int i = 0; i < sizeOfBoardd; i++) {
		const int conflicts = getConflicts(i,queens[i]) - 3;

		if (bestCount == 0 || frontConflicts == conflicts) {
			container[bestCount] = i;
			frontConflicts = conflicts;
			bestCount++;
		}
		else if (frontConflicts < conflicts) {
			frontConflicts = conflicts;
			container[0] = i;
			bestCount = 1;
		}
	}

	const int winner = rng() % bestCount;
	return std::pair<int,int>(container[winner],frontConflicts);
}

inline int getLeastConflictRow(const int queen) {
	const int currentRow = queens[queen];
	
	int bestCount = 0;
	int frontCollisions = INT32_MAX;
	for (int i = 0; i < sizeOfBoardd; i++) {
		if (currentRow == i) {
			continue;
		}
		
		const int collisions = getConflicts(queen,i);

		if (bestCount == 0 || frontCollisions == collisions) {
			container[bestCount]=i;
			frontCollisions = collisions;
			bestCount++;
		}
		else if (frontCollisions > collisions) {
			frontCollisions = collisions;
			container[0] = i;
			bestCount = 1;
		}
	}

	const int winner = rng() % bestCount;

	return container[winner];
}

void minimumConflict() {
	initializeBoard();
	int maxConflictQueen = -1;
	for (int i = 0; i <= k * sizeOfBoardd; i++) {
		const std::pair<int,int>& maxConflictQueen = getMaxConflictQueen();
		if (maxConflictQueen.second == 0) {
			return;
		}
		const int leastConflictRow = getLeastConflictRow(maxConflictQueen.first);

		updateConflictStatistics(maxConflictQueen.first, leastConflictRow);
	}

	std::cout << "restart" << std::endl;
	minimumConflict();
}

void printBoard() {
	for (int i = 0; i < sizeOfBoardd; i++) {
		for (int j = 0; j < sizeOfBoardd; j++) {
			if (queens[j] != i) std::cout << "- ";
			else std::cout << "* ";
		}
		std::cout << std::endl;
	}
}

int main() {
	std::cin >> sizeOfBoardd;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	queens = new int[sizeOfBoardd];
	rows = new int[sizeOfBoardd];
	container = new int[sizeOfBoardd];
	diagonal1 = new int[2*sizeOfBoardd-1];
	diagonal2 = new int[2*sizeOfBoardd-1];

	minimumConflict();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time consumed :" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

	//printBoard();

	return 0;
}