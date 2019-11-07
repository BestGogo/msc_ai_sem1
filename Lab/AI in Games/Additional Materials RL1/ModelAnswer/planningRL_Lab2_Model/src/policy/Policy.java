package policy;

import utils.Vector2d;
import java.util.ArrayList;

/**
 * Created by dperez on 16/09/15.
 */
public abstract class Policy {

    //Probability of taking each action attending to the policy and the state-values (v)
    public abstract double prob(double[][] value, int row, int col, Vector2d action, ArrayList<Vector2d> actions);

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of v(s).
    public abstract Vector2d sampleAction(double[][] value, int row, int col, ArrayList<Vector2d> actions);

    //Just returns a string for the name of this policy.
    public abstract String getPolicyStr();
}