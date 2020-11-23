package main

import (
	"fmt"
	commons "danielpenchev98.com/tictactoe/commons" 
	state "danielpenchev98.com/tictactoe/state" 
)

func moveMax(s *state.State, alpha int, beta int, output string) (int, int) {
	if s.IsTerminal() {
		if _, ok := s.GetWinner(); ok {
	//		fmt.Printf("%sMe terminal value reached with heurisitc %d\n",output,s.CalculateHeuristic()+1) 
			return -(s.CalculateHeuristic()+1), -1
		}
	//	fmt.Println("%sterminal value reached with heurisitc 0",output)
		return 0, -1
	}

	bestUtility := commons.MinInt
	bestMove := -1

	for _, tile := range s.GetNeightbours() {
		//fmt.Printf("%sneighbour %d\n",output,tile)

		row, col := commons.IndexToPosition(tile)		
		s.PlayMove(1, row, col)
		currUtility, _ := moveMin(s, alpha, beta,output+" ")
		s.UndoMove(1, row, col)

		if currUtility > bestUtility {
			bestUtility = currUtility
			bestMove = tile
			alpha = commons.Max(alpha, bestUtility)
		}

		if bestUtility >= beta {
			return bestUtility, bestMove
		}
	}
	return bestUtility, bestMove
}

func moveMin(s *state.State, alpha int, beta int, output string) (int, int) {
	if s.IsTerminal() {
		if _, ok := s.GetWinner(); ok {
	//		fmt.Printf("%sAI terminal value reached with heurisitc %d\n",output,s.CalculateHeuristic()+1)
			return s.CalculateHeuristic()+1, -1
		}
	//  	fmt.Println("%sterminal value reached with heurisitc 0",output)
		return 0, -1
	}

	bestUtility := commons.MaxInt
	bestMove := -1
	for _, tile := range s.GetNeightbours() {

		row, col := commons.IndexToPosition(tile)
		s.PlayMove(0,row,col)
		currUtility, _ := moveMax(s, alpha, beta,output+" ")
		s.UndoMove(0,row,col)

		if currUtility < bestUtility {
			bestUtility = currUtility
			bestMove = tile
			beta = commons.Min(beta, bestUtility)
		}

		if alpha >= bestUtility {
			return bestUtility, bestMove
		}
	}
	return bestUtility, bestMove
}

func playGame(playerNumber int) {
	game := state.CreateState()

	var x, y, mark int
	for i := 1; i < 10; i++ {
		if i%2 == playerNumber {
			fmt.Print("Its your turn\nEnter position :")
			fmt.Scanf("%d %d", &y, &x)
			mark = playerNumber
		} else {
			fmt.Println("AI's turn")
			heur, move := moveMax(game, commons.MinInt, commons.MaxInt,"")
			y, x = commons.IndexToPosition(move)
			fmt.Printf("The AI moved on position (%d,%d) with heuristic %d\n", y, x, heur)
			mark = (playerNumber+1)%2
		}
		game.PlayMove(mark, y, x)
		game.PrintHistogram()
		fmt.Printf("Game board :\n%s",game)

		if game.IsTerminal() {
			break
		}
	}

	if winner, ok := game.GetWinner(); ok {
		fmt.Printf("The winner is player%d\n",winner)
	} else {
		fmt.Printf("Its a draw")
	}
}

func main() {
	var playerNumber int
	fmt.Print("Choose player number (1-first,0-second) :")
	fmt.Scanf("%d", &playerNumber)
	fmt.Println("Game starts now")

	playGame(playerNumber)
}
