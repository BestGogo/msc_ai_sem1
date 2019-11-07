package algorithms.control;

import algorithms.planning.MonteCarlo;
import algorithms.planning.PlanningAlgorithm;
import gridworld.GridWorld;
import policy.EGreedyPolicy;
import policy.Policy;
import utils.Vector2d;

/**
 * Created by dperez on 19/09/15.
 */
public class MonteCarloControl extends Control
{
    public double alpha;
    public MonteCarloControl(GridWorld gw, double epsilon, double gamma, double alpha)
    {
        super(gw, null, gamma);
        this.alpha = alpha;
        this.policy = new EGreedyPolicy(epsilon, false);
    }

    @Override
    public void execute() {
        //v(s) = v(s) + alpha * (Return - v(s));
        double[][][] nextQValue = new double[qValues.length][qValues[0].length][gridWorld.actions.size()];

        for(int i = 0; i < qValues.length; ++i)
            for(int j = 0; j < qValues[0].length; ++j)
            {
                double val = 0.0;
                if(gridWorld.isTerminal(i,j)) {
                    val = gridWorld.getReward(i, j);
                    for (Vector2d act : gridWorld.actions) {
                        int action = gridWorld.getActionIndex(act);
                        nextQValue[i][j][action] = val;
                    }
                }else
                {
                    for(Vector2d act : gridWorld.actions)
                    {
                        double returnValue = this.getReturn(i, j, act);
                        int action = gridWorld.getActionIndex(act);
                        int nCount = incCounter(i, j, action);
                        double curQValue = this.getQValue(i, j, action);
                        val = curQValue + (1.0/ (double)nCount) * (returnValue - curQValue);

                        nextQValue[i][j][action] = val;
                    }
                }

                if(val > maxVal)
                    maxVal = val;
                if(val < minVal)
                    minVal = val;
            }

        for(int i = 0; i < qValues.length; ++i)
            for(int j = 0; j < qValues[i].length; ++j)
                System.arraycopy(nextQValue[i][j], 0, qValues[i][j], 0, gridWorld.actions.size());
    }
}
