package policy;

import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by dperez on 16/09/15.
 */
public class RandomPolicy extends Policy {

    @Override
    public double prob(double[][] value, int row, int col, Vector2d action, ArrayList<Vector2d> actions) {
        return 1.0 / (double)actions.size();
    }

    @Override
    public Vector2d sampleAction(double[][] value, int row, int col, ArrayList<Vector2d> actions) {
        int actionIdx = new Random().nextInt(actions.size());
        return actions.get(actionIdx);
    }

    @Override
    public String getPolicyStr() {
        return String.format("Random");
    }
}
