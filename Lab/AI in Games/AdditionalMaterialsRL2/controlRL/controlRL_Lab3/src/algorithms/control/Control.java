package algorithms.control;

import algorithms.planning.PlanningAlgorithm;
import gridworld.GridWorld;
import policy.GreedyPolicy;
import policy.Policy;
import utils.Vector2d;

import java.util.ArrayList;

/**
 * Created by dperez on 19/09/15.
 */
public abstract class Control extends PlanningAlgorithm{

    //Q-Values for all state-action pairs. The first two dimensions are the dimensions of the gridworld,
    // while the third one is for each one of the actions that can be taken.
    public double[][][] qValues;

    //Counts of the number of times each action has been taken from each state. It has the same dimensions as
    // the q-values array.
    public int [][][]n;

    //Creates the control algorith,
    public Control(GridWorld gw, Policy p, double gamma) {
        super(gw, p, gamma);

        //Inits the arrays of this class.
        qValues = new double[super.value.length][super.value[0].length][gw.actions.size()];
        n = new int[super.value.length][super.value[0].length][gw.actions.size()];

        //Initializes the values for the viewer.
        viewer.values = computeStateValueFromQValue();
    }

    //Gets the current q-value of a pair state (row, col) - action (action).
    public double getQValue(int row, int col, int action) {
        row = Math.max(0, Math.min(row, qValues.length - 1));
        col = Math.max(0, Math.min(col, qValues[row].length - 1));
        return qValues[row][col][action];
    }

    //Returns the number of times an action 'action' has been chosen from a given state (row, col).
    public int incCounter(int row, int col, int action) {
        row = Math.max(0, Math.min(row, qValues.length - 1));
        col = Math.max(0, Math.min(col, qValues[row].length - 1));
        n[row][col][action]++;
        return n[row][col][action];
    }

    //Gets the return from a state (row, col), starting with a first action.
    public double getReturn(int row, int col, Vector2d firstAction)
    {
        //Iterate until terminal state (or minimum gamma) and return the final reward
        double curGamma = gamma;
        int curRow = row;
        int curCol = col;
        double minGamma = 0.1;
        double acumReturn = gridWorld.getReward(curRow, curCol);
        boolean isFirst = true;


        //We don't have all the time in the world, so let's put some minimum gamma value
        while(!gridWorld.isTerminal(curRow, curCol) && curGamma >= minGamma)
        {
            Vector2d nextAction = firstAction;
            if(!isFirst)
                nextAction = policy.sampleAction(qValues, curRow, curCol, gridWorld.actions);
            isFirst = false;

            curRow = Math.max (0, Math.min(curRow + (int) nextAction.y, qValues.length-1));
            curCol = Math.max (0, Math.min(curCol + (int) nextAction.x, qValues[curRow].length-1));

            double nextReward = gridWorld.getReward(curRow, curCol);
            acumReturn += (curGamma * nextReward);
            curGamma = curGamma * gamma;
        }

        return acumReturn;
    }


    //Computes the state-action values from all state-action q-values. This is done according to the
    // equation: v = sum(actions) {p(a|s) * q(s,a)}
    public double[][] computeStateValueFromQValue()
    {
        double [][]values = new double[qValues.length][qValues[0].length];
        for(int i = 0; i < qValues.length; ++i)
            for(int j = 0; j < qValues[0].length; ++j)
            {
                double val = 0.0;
                for(Vector2d act : gridWorld.actions)
                {
                    int action = gridWorld.getActionIndex(act);
                    val += new GreedyPolicy().prob(qValues, i, j, act, gridWorld.actions) * qValues[i][j][action];
                }
                values[i][j] = val;
            }

        return values;
    }



    public double prob(int row, int col, Vector2d action, ArrayList<Vector2d> actions)
    {
        double value[][] = computeStateValueFromQValue();
        return policy.prob(value, row, col, action, actions);
    }

    public void draw()
    {
        //If we are in control, compute V from Q and return.
        double value[][] = computeStateValueFromQValue();
        viewer.paint(minVal, maxVal, value, qValues);
    }

}
