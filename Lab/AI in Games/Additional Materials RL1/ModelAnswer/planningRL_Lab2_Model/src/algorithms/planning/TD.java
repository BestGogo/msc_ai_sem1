package algorithms.planning;

import gridworld.GridWorld;
import policy.Policy;
import utils.Vector2d;

/**
 * This is TD
 * Created by dperez on 16/09/15.
 */
public class TD extends PlanningAlgorithm
{
    public double alpha;

    public TD(GridWorld gw, Policy p, double gamma, double alpha) {
        super(gw, p, gamma);

        this.alpha = alpha;
        for(int i = 0; i < value.length; ++i)
            for(int j = 0; j < value[0].length; ++j)
                this.value[i][j] = 0.0; //Values all 0.0
    }

    @Override
    //Generates the next set of state-value values v(s). It should iterate through all states in the gridworld
    //and update the value of v(s) according to the TD update: v(s) = v(s) + alpha * (Return + gamma * v(s') - v(s));
    public void execute() {

        //v(s) = v(s) + alpha * (Return + gamma * v(s') - v(s));
        double[][] nextValue = new double[value.length][value[0].length];

        for(int i = 0; i < value.length; ++i)
            for(int j = 0; j < value[0].length; ++j)
            {
                double val = 0.0;
                if(gridWorld.isTerminal(i,j))
                    val = gridWorld.getReward(i,j);
                else
                {
                    double curV = this.getValue(i,j);

                    Vector2d nextAction = policy.sampleAction(value, i, j, gridWorld.actions);
                    int row = Math.max(0, Math.min(i + (int) nextAction.y, value.length - 1));
                    int col = Math.max(0, Math.min(j + (int) nextAction.x, value[i].length - 1));
                    double nextReward = gridWorld.getReward(row, col);

                    double nextV = this.getValue(row, col);

                    double tdTarget = nextReward + gamma * nextV;
                    double tdError = tdTarget - curV;
                    val = curV + alpha * (tdError);
                }

                nextValue[i][j] = val;

                if(val > maxVal)
                    maxVal = val;
                if(val < minVal)
                    minVal = val;
            }

        for(int i = 0; i < value.length; ++i)
            System.arraycopy(nextValue[i], 0, value[i], 0, value.length);

    }

}
