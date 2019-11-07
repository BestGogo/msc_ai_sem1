package algorithms.control;

import gridworld.GridWorld;
import policy.EGreedyPolicy;
import utils.Vector2d;

/**
 * Created by dperez on 19/09/15.
 */
public class QLearning extends Control
{
    public double alpha;
    public QLearning(GridWorld gw, double epsilon, double gamma, double alpha)
    {
        super(gw, null, gamma);
        this.alpha = alpha;
        this.policy = new EGreedyPolicy(epsilon, true);
    }

    @Override
    public void execute() {
        //v(s) = v(s) + alpha * (R - v(s));
        double[][][] nextQValue = new double[qValues.length][qValues[0].length][gridWorld.actions.size()];
        //Copy q-values.
        for(int i = 0; i < qValues.length; ++i)
            for(int j = 0; j < qValues[i].length; ++j)
                System.arraycopy(qValues[i][j], 0, nextQValue[i][j], 0, gridWorld.actions.size());


        for(int i = 0; i < qValues.length; ++i)
            for(int j = 0; j < qValues[0].length; ++j)
            {
                if(gridWorld.isTerminal(i,j)) {
                    double val = gridWorld.getReward(i, j);
                    if (val > maxVal) maxVal = val;
                    if (val < minVal) minVal = val;
                    for(Vector2d act : gridWorld.actions) {
                        int action = gridWorld.getActionIndex(act);
                        nextQValue[i][j][action] = val;
                    }
                }else{
                    qlearning(nextQValue, i, j);
                }
            }


        for(int i = 0; i < qValues.length; ++i)
            for(int j = 0; j < qValues[i].length; ++j)
                System.arraycopy(nextQValue[i][j], 0, qValues[i][j], 0, gridWorld.actions.size());
    }


    private void qlearning(double[][][] nextQValue, int s_row, int s_col)
    {

        //TODO 7: Implement the update of Q(s,a), using the Q-Learning algorithm, for the state (s_row, s_col). Follow the next steps:

        //1. Choose 'a' according to policy.
        Vector2d a = null; //?

        //This is the index of that action 'a' in the array nextQValue[][]
        int aIdx = gridWorld.getActionIndex(a);

        //2. Take action 'a', observe 'R' and S'

        //3. Get the original Q(s,a) (Q)

        //4. Get the VALUE that maximizes Q on the next state:  max_{a} Q(s',a) (use getMaxQValue(), the function below).

        //5. Update q(s,a) using: Q <- Q + alpha * (R + gamma * maxQ(s',a) - Q)
        double val = 0.0;


        //Leave this as it is:
        nextQValue[s_row][s_col][aIdx] = val;
    }


    public double getMaxQValue(int row, int col)
    {
        double maxValue = -Double.MAX_VALUE;
        for(Vector2d act : gridWorld.actions)
        {
            int actionIdx = GridWorld.getActionIndex(act);
            double val = qValues[row][col][actionIdx];

            if(val > maxValue)
            {
                maxValue = val;
            }
        }
        return maxValue;
    }
}
