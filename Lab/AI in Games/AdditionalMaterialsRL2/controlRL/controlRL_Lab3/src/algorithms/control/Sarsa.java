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
        //TODO 6: Implement the update of Q(s,a), using the sarsa algorithm, for the state (s_row, s_col). Follow the next steps:


        //1. Choose 'a' according to policy.
        Vector2d a = null; //?

        //This is the index of that action 'a' in the array nextQValue[][]
        int aIdx = gridWorld.getActionIndex(a);

        //2. Take action 'a', observe 'R' and S'

        //3. Choose a' from s' using policy.

        //4. Get the original Q(s,a) (Q)

        //5. Get the next Q(s',a') (Q')

        //6. Apply: Q <- Q + alpha * (R + gamma * Q' - Q)
        double val = 0.0; //Change this.


        //Leave this as it is:
        nextQValue[s_row][s_col][aIdx] = val;
    }
}
