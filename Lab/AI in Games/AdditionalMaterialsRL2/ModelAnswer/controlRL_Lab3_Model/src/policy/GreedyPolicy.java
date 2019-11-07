package policy;

import gridworld.GridWorld;
import utils.Vector2d;

import java.util.ArrayList;

/**
 * Created by dperez on 16/09/15.
 */
public class GreedyPolicy extends Policy {

    //Probability of taking each action attending to the policy and the state-values (v)
    @Override
    public double prob(double[][] value, int row, int col, Vector2d action, ArrayList<Vector2d> actions) {

        double maxValue = -Double.MAX_VALUE;
        ArrayList<Vector2d> bestActions = new ArrayList<>();
        int countBest = 0;

        for(Vector2d act : actions)
        {
            int newRow = Math.max (0, Math.min(row + (int) act.y, value.length-1));
            int newCol = Math.max (0, Math.min(col + (int) act.x, value[newRow].length-1));
            double val = value[newRow][newCol];

            if(val > maxValue)
            {
                maxValue = val;
                bestActions.clear();
                bestActions.add(act);
                countBest = 1;
            }else if(val == maxValue)
            {
                countBest++;
                bestActions.add(act);
            }
        }

        if(bestActions.contains(action))
        {
            return 1.0 / (double)countBest;
        }

        return 0.0;
    }

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of v(s).
    @Override
    public Vector2d sampleAction(double[][] value, int row, int col, ArrayList<Vector2d> actions) {
        return sample(value, row, col, actions);
    }

    //Probability of taking each action attending to the policy and the state-action values (q)
    @Override
    public double prob(double[][][] qValue, int row, int col, Vector2d action, ArrayList<Vector2d> actions) {
        double maxValue = -Double.MAX_VALUE;
        ArrayList<Vector2d> bestActions = new ArrayList<>();
        int countBest = 0;

        for(Vector2d act : actions)
        {
            int actionIdx = GridWorld.getActionIndex(act);
            double val = qValue[row][col][actionIdx];

            if(val > maxValue)
            {
                maxValue = val;
                bestActions.clear();
                bestActions.add(act);
                countBest = 1;
            }else if(val == maxValue)
            {
                countBest++;
                bestActions.add(act);
            }
        }

        if(bestActions.contains(action))
        {
            return 1.0 / (double)countBest;
        }

        return 0.0;
    }

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of q(s,a).
    @Override
    public Vector2d sampleAction(double[][][] value, int row, int col, ArrayList<Vector2d> actions) {
        return sample(value, row, col, actions);
    }

    @Override
    public String getPolicyStr() {
        return String.format("Greedy");
    }



}