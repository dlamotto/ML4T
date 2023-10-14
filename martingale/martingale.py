""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Danielle LaMotto 	  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: dlamotto3 	   		  		 		  		  		    	 		 		   		 		  
GT ID: 903951588 		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def author():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    return "dlamotto3"  # replace tb34 with your Georgia Tech username.
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def gtid():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    return 903951588  # replace with your GT ID number
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		  		 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    result = False  		  	   		  		 		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		  		 		  		  		    	 		 		   		 		  
        result = True  		  	   		  		 		  		  		    	 		 		   		 		  
    return result  		  	   		  		 		  		  		    	 		 		   		 		  


def test_code(num_of_episodes, bankroll_allowed=False):
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    win_prob = 0.47  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments
    # Note: If the method get_spin_result returns true that means it rolled a black, if false it rolled a red
    # Note: when you win reset bet amount to $1, if you lose double down

    # 2d array with 1000 columns and rows = # of episodes wanted
    episodes_array = np.empty((num_of_episodes, 1000), dtype=np.double)
    # initialize each episode to begin with 0
    episodes_array[:, 0] = 0
    count_reach = 0
    for episode in range(num_of_episodes):  # running # of episodes
        episode_winnings = 0  # start the winnings from 0 each episode
        bet_amount = 1

        for spin in range(1, 10000):
            if -256 < episode_winnings < 80:
                won = get_spin_result(win_prob)  # probability of landing on a black
                count_reach+= 1
                if won:  # if true(landed on a black)
                    episode_winnings += bet_amount  # add for a win
                    bet_amount=1
                else:  # if false(landed on a red)
                    episode_winnings -= bet_amount  # subtract for a loss
                    # when the next bet = $N, but you only have $M (where M<N); only bet $M.
                    if episode_winnings < bet_amount and bankroll_allowed:
                        bet_amount = abs(episode_winnings)
                    bet_amount = bet_amount * 2  # double down
            else:
                # when the winnings <= -256 and bankroll is allowed, stop betting and keep the value at -256
                if episode_winnings <= -256:
                    if bankroll_allowed:
                        episode_winnings = -256  # Stop betting and set episode_winnings to -256
                        episodes_array[episode, spin:] = -256  # Fill the rest of the episode with -256
                        break
                    else:
                        continue  # If bankroll_allowed is False, continue betting
                else:
                    episode_winnings = 80  # Stop betting and set episode_winnings to 80
                    episodes_array[episode, spin:] = 80  # Fill the rest of the episode with 80
                    break
            episodes_array[episode, spin] = episode_winnings

    #print(count_reach)
    # prob = np.mean(episodes_array, axis=1)
    # #ans = prob/1000
    # print(prob)

    # different plots
    # figure 1
    if num_of_episodes == 10 and not bankroll_allowed:
        for each_episode in range(num_of_episodes):
            plt.plot(episodes_array[each_episode, :300])  # plots up to 300 spins
        plt.xlabel('Spins')
        plt.ylabel('Winnings')
        plt.title("Figure 1")
        plt.axis([0, 300, -256, 100])
        plt.legend([f'Episode {i + 1}' for i in range(10)])
        plt.savefig('images/figure1.png')
        plt.close()
        # plt.show()
    #
    # # figure 2
    if num_of_episodes == 1000 and not bankroll_allowed:
        spins_means = np.mean(episodes_array, axis=0)
        #expected_val = np.mean(spins_means)
        #print(spins_means)
        spins_stds = np.std(episodes_array, axis=0, ddof=0)  # ddof = 0 is for popular standard deviation
        pos_std_devs = spins_means + spins_stds
        neg_std_devs = spins_means - spins_stds
        plt.plot(spins_means[:300], label='Mean')  # plots up to 300 spins
        plt.plot(pos_std_devs[:300], label='Upper Std Dev')
        plt.plot(neg_std_devs[:300], label='Lower Std Dev')
        plt.xlabel('Spins')
        plt.ylabel('Winnings')
        plt.title("Figure 2")
        plt.axis([0, 300, -256, 100])
        plt.legend()
        plt.savefig('images/figure2.png')
        plt.close()
        # plt.show()
    #
    # figure 3
    if num_of_episodes == 1000 and not bankroll_allowed:
        winnings_medians = np.median(episodes_array, axis=0)
        winnings_stds = np.std(episodes_array, axis=0, ddof=0)  # ddof = 0 is for popular standard deviation
        pos_std_devs = winnings_medians + winnings_stds
        neg_std_devs = winnings_medians - winnings_stds
        plt.plot(winnings_medians[:300], label='Median')
        plt.plot(pos_std_devs[:300], label='Upper Std Dev')
        plt.plot(neg_std_devs[:300], label='Lower Std Dev')
        plt.xlabel('Spins')
        plt.ylabel('Winning')
        plt.title("Figure 3")
        plt.axis([0, 300, -256, 100])
        plt.legend()
        plt.savefig('images/figure3.png')
        plt.close()
        # plt.show()
    #
    # figure 4
    if num_of_episodes == 1000 and bankroll_allowed:
        bankroll_spins_means = np.mean(episodes_array, axis=0)
        bankroll_spins_stds = np.std(episodes_array, axis=0, ddof=0)  # ddof = 0 is for popular standard deviation
        pos_std_devs = bankroll_spins_means + bankroll_spins_stds
        neg_std_devs = bankroll_spins_means - bankroll_spins_stds
        plt.plot(bankroll_spins_means, label='Mean')  # plots up to 300 spins
        plt.plot(pos_std_devs[:300], label='Upper Std Dev')
        plt.plot(neg_std_devs[:300], label='Lower Std Dev')
        plt.xlabel('Spins with a Bankroll')
        plt.ylabel('Winnings with a Bankroll')
        plt.title("Figure 4")
        plt.axis([0, 300, -256, 100])
        plt.legend()
        plt.savefig('images/figure4.png')
        plt.close()
        # plt.show()

    # figure 5
    if num_of_episodes == 1000 and bankroll_allowed:
        bankroll_winnings_medians = np.median(episodes_array, axis=0)
        bankroll_winnings_stds = np.std(episodes_array, axis=0, ddof=0)  # ddof = 0 is for popular standard deviation
        pos_std_devs = bankroll_winnings_medians + bankroll_winnings_stds
        neg_std_devs = bankroll_winnings_medians - bankroll_winnings_stds
        plt.plot(bankroll_winnings_medians[:300], label='Median')
        plt.plot(pos_std_devs[:300], label='Upper Std Dev')
        plt.plot(neg_std_devs[:300], label='Lower Std Dev')
        plt.xlabel('Spins with a Bankroll')
        plt.ylabel('Winnings with a Bankroll')
        plt.title("Figure 5")
        plt.axis([0, 300, -256, 100])
        plt.legend()
        plt.savefig('images/figure5.png')
        plt.close()
        # plt.show()


if __name__ == "__main__":
    test_code(10)  # 10 episodes, bankroll set to an unattainable number for experiment 1
    test_code(1000)  # 1000 episodes, the mean and median
    test_code(1000, bankroll_allowed=True)  # 1000 episodes, bankroll, the mean and median
