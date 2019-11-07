package algorithms.planning;

import gridworld.GridWorld;
import policy.Policy;
import utils.Vector2d;

/**
 * Created by dperez on 22/08/15.
 */
public class ValueIteration extends PlanningAlgorithm {

    //Creates the value iteration class.
    public ValueIteration(GridWorld gw, double gamma)
    {
        super(gw, null, gamma); //no need for gamma

        //Initialization
        for(int i = 0; i < value.length; ++i)
            for(int j = 0; j < value[0].length; ++j)
                //Initialization doesn't matter:
                //this.value[i][j] = rnd.nextDouble() - 0.5; //Values [-0.5 .. 0.5]
                this.value[i][j] = 0.0; //Values all 0.0
    }


    @Override
    //Generates the next set of state-value values v(s). It should iterate through all states in the gridworld
    //and update the value of v(s) according to the value iteration update: v(s) = MAX{a in A} reward + gamma * v(s')
    public void execute()
    {
        double[][] nextValue = new double[value.length][value[0].length];

        for(int i = 0; i < value.length; ++i)
            for(int j = 0; j < value[0].length; ++j)
            {
                double val = 0.0;
                if(gridWorld.isTerminal(i,j))
                    val = gridWorld.getReward(i,j);
                else
                {
                    Vector2d action = getActionWithMaxV(i, j);
                    double nextReward = gridWorld.getReward(i, j);
                    val = nextReward + gamma * this.getValue(i, j, action);
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

    //Returns the action that, after being applied, takes the MDP to the state
    //with a highest value v(s)
    private Vector2d getActionWithMaxV(int row, int col)
    {
        double maxValue = -Double.MAX_VALUE;
        Vector2d action = null;

        for(Vector2d act : gridWorld.actions)
        {
            int newRow = Math.max (0, Math.min(row + (int) act.y, value.length-1));
            int newCol = Math.max (0, Math.min(col + (int) act.x, value[newRow].length-1));
            double v = this.getValue(newRow, newCol);

            if(v > maxValue)
            {
                maxValue = v;
                action = act;
            }
        }

        return action;
    }


}
