package algorithms.control;

import algorithms.planning.MonteCarlo;
import algorithms.planning.PlanningAlgorithm;
import algorithms.planning.TD;
import gridworld.GridWorld;
import policy.EGreedyPolicy;
import policy.Policy;
import policy.RandomPolicy;
import utils.Vector2d;

/**
 * Created by dperez on 19/09/15.
 */
public class Sarsa extends Control
{
    public double alpha;
    public Sarsa(GridWorld gw, double epsilon, double gamma, double alpha)
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
                    sarsa(nextQValue, i, j);
                }
            }


        for(int i = 0; i < qValues.length; ++i)
            for(int j = 0; j < qValues[i].length; ++j)
                System.arraycopy(nextQValue[i][j], 0, qValues[i][j], 0, gridWorld.actions.size());
    }


    private void sarsa(double[][][] nextQValue, int s_row, int s_col)
    {
        //Choose 'a' according to policy.
        Vector2d a = policy.sampleAction(qValues, s_row, s_col, gridWorld.actions);

        //Take action 'a', observe 'R' and S'
        double R = gridWorld.getReward(s_row, s_col);
        int sP_row = Math.max(0, Math.min(s_row + (int) a.y, qValues.length - 1));
        int sP_col = Math.max(0, Math.min(s_col + (int) a.x, qValues[s_row].length - 1));

        //Choose a' from s' using policy.
        Vector2d aPrime = policy.sampleAction(qValues, sP_row, sP_col, gridWorld.actions);

        //Original Q(s,a)
        int aIdx = gridWorld.getActionIndex(a);
        double curQValue = this.getQValue(s_row, s_col, aIdx);

        //Next Q(s',a')
        int aPrimeIdx = gridWorld.getActionIndex(aPrime);
        double nextQA = this.getQValue(sP_row, sP_col, aPrimeIdx);

        //Q <- Q + alpha * (R + gamma * Q' - Q)
        double val = curQValue + alpha * (R + gamma * nextQA - curQValue);
        if (val > maxVal)
            maxVal = val;
        if (val < minVal)
            minVal = val;
        nextQValue[s_row][s_col][aIdx] = val;
    }
}
