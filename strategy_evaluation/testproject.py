from ManualStrategy import report_ms
from StrategyLearner import report_sl
from experiment1 import report_1
from experiment2 import report_2




def testproject():
    print("Running Manual Strategy \n")
    report_ms()
    print("\n")
    print("Running Strategy Learner \n")
    report_sl()
    print("\n")
    print("Running Experiment 1 \n")
    report_1()
    print("\n")
    print("Running Experiment 2 \n")
    report_2()
    print("\n")


if __name__ == "__main__":
    testproject()
