package main

import (
	"fmt"
	commons "danielpenchev98.com/tictactoe/commons" 
	state "danielpenchev98.com/tictactoe/state" 
)

func moveMax(s *state.State, alpha int, beta int, output string) (int, int) {
	if s.IsWon(){
		return -(s.CalculateHeuristic()+1), -1
	} else if s.IsTerminal() {
		return 0, -1
	}

	bestUtility := commons.MinInt
	bestMove := -1

	neightbours := s.GetNeightbours()

	for _, tile := range neightbours {

		row, col := commons.IndexToPosition(tile)
		s.PlayMove(1, row, col)

		currUtility, _ := moveMin(s, alpha, beta, output+" ")

		if currUtility > bestUtility {
			bestUtility = currUtility
			bestMove = tile
			alpha = commons.Max(alpha, bestUtility)
		}

		s.UndoMove(1, row, col)

		if bestUtility >= beta {
			return bestUtility, bestMove
		}
	}
	return bestUtility, bestMove
}

func moveMin(s *state.State, alpha int, beta int, output string) (int, int) {
	if  s.IsWon(){
		return s.CalculateHeuristic()+1, -1
	}
	if s.IsTerminal() {
		return 0, -1
	}

	neightbours := s.GetNeightbours()

	bestUtility := commons.MaxInt
	bestMove := -1
	for _, tile := range neightbours {

		row, col := commons.IndexToPosition(tile)
		s.PlayMove(0, row, col)

		currUtility, _ := moveMax(s, alpha, beta, output+" ")
		if currUtility < bestUtility {
			bestUtility = currUtility
			bestMove = tile
			beta = commons.Min(beta, bestUtility)
		}

		s.UndoMove(0, row, col)

		if alpha >= bestUtility {
			return bestUtility, bestMove
		}
	}
	return bestUtility, bestMove
}


//human is player2
func firstMoveAI() {

	game := state.CreateState()

	var x, y, mark int
	for i := 0; i < 9; i++ {
		if i%2 == 0 {
			fmt.Println("AI's turn")
			heur, move := moveMax(game, commons.MinInt, commons.MaxInt, "")
			y = move / 3
			x = move % 3
			fmt.Printf("The AI moved on position (%d,%d) with heuristic %d\n", x, y, heur)
			mark = 1
		} else {
			fmt.Print("Its your turn\nEnter position :")
			fmt.Scanf("%d %d", &y, &x)
			mark = 0
		}
		game.PlayMove(mark, x, y)
		fmt.Printf("Game board :\n%s",game)

		if game.IsWon(){
			fmt.Printf("The winner is player%d\n",(i%2)+1)
			break;
		}
	}
}

func main() {
	var playerNumber int
	fmt.Scanf("%d", &playerNumber)

	fmt.Println("Game starts now")

	firstMoveAI()
}
