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
        //Choose 'a' according to policy.
        Vector2d a = policy.sampleAction(qValues, s_row, s_col, gridWorld.actions);

        //Take action 'a', observe 'R' and S'
        double R = gridWorld.getReward(s_row, s_col);
        int sP_row = Math.max(0, Math.min(s_row + (int) a.y, qValues.length - 1));
        int sP_col = Math.max(0, Math.min(s_col + (int) a.x, qValues[s_row].length - 1));

        //Original Q(s,a)
        int aIdx = gridWorld.getActionIndex(a);
        double curQValue = this.getQValue(s_row, s_col, aIdx);

        //Get the VALUE that maximizes Q on the next state:  max_{a} Q(s',a)
        double maxQA = this.getMaxQValue(sP_row, sP_col);

        //Q <- Q + alpha * (R + gamma * maxQ(s',a) - Q)
        double val = curQValue + alpha * (R + gamma * maxQA - curQValue);
        if (val > maxVal)
            maxVal = val;
        if (val < minVal)
            minVal = val;
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
