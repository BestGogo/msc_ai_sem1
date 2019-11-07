package players.group_AA;

import core.GameState;
import players.heuristics.AdvancedHeuristic;
import players.heuristics.CustomHeuristic;
import players.heuristics.StateHeuristic;
import utils.ElapsedCpuTimer;
import utils.Types;
import utils.Utils;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

public class SingleTreeNode
{
    public Params params;

    private SingleTreeNode parent;
    private SingleTreeNode[] children;
    private double totValue;
    private double totAMAFValue;
    private ArrayList<Double> samples;
    private int nVisits;
    private int nAMAFVisits;
    private Random m_rnd;
    private int m_depth;
    private double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    private double[] AMAFbounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    private int childIdx;
    private int fmCallsCount;
    public int c=-1;
    private int num_actions;
    private Types.ACTIONS[] actions;
    private double raveAlpha;

    private GameState rootState;
    private StateHeuristic rootStateHeuristic;

    SingleTreeNode(Params p, Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this(p, null, -1, rnd, num_actions, actions, 0, null);
    }

    private SingleTreeNode(Params p, SingleTreeNode parent, int childIdx, Random rnd, int num_actions,
                           Types.ACTIONS[] actions, int fmCallsCount, StateHeuristic sh) {
        this.params = p;
        this.fmCallsCount = fmCallsCount;
        this.parent = parent;
        this.m_rnd = rnd;
        this.num_actions = num_actions;
        this.actions = actions;
        children = new SingleTreeNode[num_actions];
        samples = new ArrayList<>();
        totValue = 0.0;
        this.childIdx = childIdx;
        if(parent != null) {
            m_depth = parent.m_depth + 1;
            this.rootStateHeuristic = sh;
        }
        else
            m_depth = 0;

    }

    void setRootGameState(GameState gs)
    {
        this.rootState = gs;
        if (params.heuristic_method == params.CUSTOM_HEURISTIC)
            this.rootStateHeuristic = new CustomHeuristic(gs);
        else if (params.heuristic_method == params.ADVANCED_HEURISTIC) // New method: combined heuristics// New method: combined heuristics
            this.rootStateHeuristic = new AdvancedHeuristic(gs, m_rnd);
        else if (params.heuristic_method == params.HEURISTICSAA)
            this.rootStateHeuristic = new HeuristicsAA(gs, m_rnd);
    }


    void mctsSearch(ElapsedCpuTimer elapsedTimer) {

        double avgTimeTaken;
        double acumTimeTaken = 0;
        long remaining;
        int numIters = 0;

        int remainingLimit = 5;
        boolean stop = false;

        while(!stop){

            GameState state = rootState.copy();
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            SingleTreeNode selected = treePolicy(state, numIters);
            double delta = selected.rollOut(state);
            if (!params.AMAFPolicyEnable)
                backUp(selected, delta);
            else
                backUpAMAF(selected,delta);

            //Stopping condition
            if(params.stop_type == params.STOP_TIME) {
                numIters++;
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;
                avgTimeTaken  = acumTimeTaken/numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            }else if(params.stop_type == params.STOP_ITERATIONS) {
                numIters++;
                stop = numIters >= params.num_iterations;
            }else if(params.stop_type == params.STOP_FMCALLS)
            {
                fmCallsCount+=params.rollout_depth;
                stop = (fmCallsCount + params.rollout_depth) > params.num_fmcalls;
            }
        }
//        System.out.println(" ITERS " + numIters);
    }

    private SingleTreeNode treePolicy(GameState state , int numIterations) {

        SingleTreeNode cur = this;

        while (!state.isTerminal() && cur.m_depth < params.rollout_depth)
        {
            if (cur.notFullyExpanded()) {
                return cur.expand(state);

            } else {
                if (!params.AMAFPolicyEnable) {
                    switch (params.policyType){
                        case "uct" : cur = cur.uct(state); break;
                        case "uct2Tuned": cur = cur.uct2Tuned(state); break;
                        case "uctBayesian": cur = cur.uctBayesian(state); break;
                        case "epsilonGreedy": cur = cur.epsilonGreedy(state); break;
                        case "epsilonGreedyDecay": cur = cur.epsilonGreedyDecay(state); break;
                        default:
                            System.out.println("WARNING: Invalid agent policy " + params.policyType + " for UCT and epsilon type");
                    }
                }else{
                    switch (params.policyType) {
                        case "AMAF": cur = cur.AMAF(state);
                        case "alphaAMAF": cur = cur.alphaAMAF(state, params.AMAPAplha);
                        case "cutOFFAMAF": cur = cur.cutOFFAMAF(state, numIterations);
                        case "grave": cur = cur.grave(state);
                        default:
                            System.out.println("WARNING: Invalid agent policy " + params.policyType + " for AMAF and other implemation of AMAF");

                    }
                }

            }
        }
        return cur;
    }



    private SingleTreeNode expand(GameState state) {

        int bestAction = 0;
        double bestValue = -1;

        for (int i = 0; i < children.length; i++) {
            double x = m_rnd.nextDouble();
            if (x > bestValue && children[i] == null) {
                bestAction = i;
                bestValue = x;
            }
        }

        //Roll the state
        roll(state, actions[bestAction]);

        SingleTreeNode tn = new SingleTreeNode(params,this,bestAction,this.m_rnd,num_actions,
                actions, fmCallsCount, rootStateHeuristic);
        children[bestAction] = tn;
        return tn;
    }

    private void roll(GameState gs, Types.ACTIONS act)
    {
        //Simple, all random first, then my position.
        int nPlayers = 4;
        Types.ACTIONS[] actionsAll = new Types.ACTIONS[4];
        int playerId = gs.getPlayerId() - Types.TILETYPE.AGENT0.getKey();

        for(int i = 0; i < nPlayers; ++i)
        {
            if(playerId == i)
            {
                actionsAll[i] = act;
            }else {
                int actionIdx = m_rnd.nextInt(gs.nActions());
                actionsAll[i] = Types.ACTIONS.all().get(actionIdx);
            }
        }

        gs.next(actionsAll);

    }

    private SingleTreeNode uct(GameState state) {
        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;
        for (SingleTreeNode child : this.children)
        {
            double hvVal = child.totValue;
            double childValue =  hvVal / (child.nVisits + params.epsilon);

            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);

            double uctValue = childValue +
                    params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));

            uctValue = Utils.noise(uctValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (uctValue > bestValue) {
                selected = child;
                bestValue = uctValue;
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        roll(state, actions[selected.childIdx]);

        return selected;
    }
    private double sd(ArrayList<Double> samples, double meanValue)
    {
        // Step 1:
        double mean = meanValue;
        double temp = 0;
        for (int i = 0; i < samples.size(); i++)
        {
            Double val = samples.get(i);
            // Step 2:
            double squrDiffToMean = Math.pow(val - mean, 2);
            // Step 3:
            temp += squrDiffToMean;
        }

        // Step 4:
        double meanOfDiffs = (double) temp / (double) (samples.size());

        // Step 5:
        return Math.sqrt(meanOfDiffs);
    }

    private SingleTreeNode uct2Tuned(GameState state) {
        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;
        for (SingleTreeNode child : this.children)
        {
            double hvVal = child.totValue;
            ArrayList<Double> childValues = child.samples;

            double childValue =  hvVal / (child.nVisits + params.epsilon);
            double meanValue =  hvVal / (child.nVisits);

            double variance = Math.pow(sd(childValues,meanValue),2);
            double v_jn_j = variance +
                    params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));

            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);

            double uctValue = childValue +
                    Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon)*Math.min(.25,v_jn_j));

            uctValue = Utils.noise(uctValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (uctValue > bestValue) {
                selected = child;
                bestValue = uctValue;
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        roll(state, actions[selected.childIdx]);

        return selected;
    }
    private SingleTreeNode epsilonGreedy(GameState state) {
        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;

        c = c+1;
        if(c>=this.nVisits){
            c=0;
        }
        if(c<=params.epsilon*this.nVisits){
            selected = this.children[m_rnd.nextInt(this.children.length)];
            System.out.println("Random selected");
        }
        else {
            for (SingleTreeNode child : this.children) {
                double hvVal = child.totValue; //rewards
                ArrayList<Double> childValues = child.samples;

//                double childValue = hvVal / (child.nVisits + params.epsilon);
                double meanValue = hvVal / (child.nVisits + params.epsilon );

                double q_k_1 = meanValue + (1/ (child.nVisits + params.epsilon  + 1)) * (Utils.sumArrayList(childValues)/childValues.size() - meanValue);
                q_k_1 = Utils.noise(q_k_1, params.epsilon, this.m_rnd.nextDouble());

                System.out.println("Greedy selected");
                if (q_k_1 > bestValue) {
                    selected = child;
                    bestValue = q_k_1;
                }
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:

        roll(state, actions[selected.childIdx]);

        return selected;
    }
    private SingleTreeNode epsilonGreedyDecay(GameState state) {
        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;

        c = c+1;
        if(c>=this.nVisits){
            c=0;
        }

        if(c<=params.epsilon*this.nVisits){
            selected = this.children[m_rnd.nextInt(this.children.length)];
        }
        else {
            for (SingleTreeNode child : this.children) {
                double hvVal = child.totValue; //rewards
                ArrayList<Double> childValues = child.samples;

//                double childValue = hvVal / (child.nVisits + params.epsilon);
                double meanValue = hvVal / (child.nVisits + params.epsilon );

                double q_k_1 = meanValue + (1/ (child.nVisits + params.epsilon  + 1)) * (Utils.sumArrayList(childValues)/childValues.size() - meanValue);
                q_k_1 = Utils.noise(q_k_1, params.epsilon, this.m_rnd.nextDouble());
                if (q_k_1 > bestValue) {
                    selected = child;
                    bestValue = q_k_1;
                }
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:

        roll(state, actions[selected.childIdx]);
        params.factor = params.factor * 10;
        params.decayEpsilon = params.decayEpsilon/params.factor;
        return selected;
    }
    private SingleTreeNode uctBayesian(GameState state) {
        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;
        for (SingleTreeNode child : this.children)
        {
            double hvVal = child.totValue;
            ArrayList<Double> childValues = child.samples;
            double childValue =  hvVal / (child.nVisits + params.epsilon);
            double meanValue =  hvVal / (child.nVisits);

            double variance = Math.pow(sd(childValues,meanValue),2);
            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);

            double uctValue = childValue + (
                    params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon))) * variance;

            uctValue = Utils.noise(uctValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (uctValue > bestValue) {
                selected = child;
                bestValue = uctValue;
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        roll(state, actions[selected.childIdx]);

        return selected;
    }

    private SingleTreeNode AMAF(GameState state) {
        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;
        for (SingleTreeNode child : this.children)
        {
            double hvVal = child.totValue;
            double childValue =  hvVal / (child.nVisits + params.epsilon);

            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);

            double uctValue = childValue +
                    params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));

            uctValue = Utils.noise(uctValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            double AMAFhvVal = child.totAMAFValue;
            double AMAFchildValue =  AMAFhvVal / (child.nAMAFVisits + params.epsilon);

            AMAFchildValue = Utils.normalise(AMAFchildValue, AMAFbounds[0], AMAFbounds[1]);

            double AMAFValue = AMAFchildValue +
                    params.K * Math.sqrt(Math.log(child.nAMAFVisits + 1) / (child.nAMAFVisits + params.epsilon));

            AMAFValue = Utils.noise(AMAFValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (AMAFValue > bestValue) {
                selected = child;
                bestValue = AMAFValue;
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        roll(state, actions[selected.childIdx]);

        return selected;
    }

    private SingleTreeNode alphaAMAF(GameState state ,double alpha) {
        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;
        for (SingleTreeNode child : this.children)
        {
            double hvVal = child.totValue;
            double childValue =  hvVal / (child.nVisits + params.epsilon);

            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);

            double uctValue = childValue +
                    params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));

            uctValue = Utils.noise(uctValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            double AMAFhvVal = child.totAMAFValue;
            double AMAFchildValue =  AMAFhvVal / (child.nAMAFVisits + params.epsilon);

            AMAFchildValue = Utils.normalise(AMAFchildValue, AMAFbounds[0], AMAFbounds[1]);

            double AMAFValue = AMAFchildValue +
                    params.K * Math.sqrt(Math.log(child.nAMAFVisits + 1) / (child.nAMAFVisits + params.epsilon));

            AMAFValue = Utils.noise(AMAFValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            double alphaAMAFValue = ((alpha * AMAFValue ) + ((1- alpha) * uctValue) );
            alphaAMAFValue = Utils.noise(alphaAMAFValue, params.epsilon, this.m_rnd.nextDouble());

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (alphaAMAFValue > bestValue) {
                selected = child;
                bestValue = alphaAMAFValue;
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        roll(state, actions[selected.childIdx]);

        return selected;
    }

    private SingleTreeNode grave(GameState state) {
        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;
        for (SingleTreeNode child : this.children)
        {
            double hvVal = child.totValue;
            double childValue =  hvVal / (child.nVisits + params.epsilon);

            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);

            double AMAFhvVal = child.totAMAFValue;
            double AMAFchildValue =  AMAFhvVal / (child.nAMAFVisits + params.epsilon);

            AMAFchildValue = Utils.normalise(AMAFchildValue, AMAFbounds[0], AMAFbounds[1]);

            double AMAFValue = AMAFchildValue +
                    params.K * Math.sqrt(Math.log(child.nAMAFVisits + 1) / (child.nAMAFVisits + params.epsilon));

            AMAFValue = Utils.noise(AMAFValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            double betam = child.nAMAFVisits / ((child.nAMAFVisits+child.nVisits)*(child.nAMAFVisits*child.nVisits));

            double graveValue = ((1-betam)*childValue) + (betam*AMAFValue);
            graveValue = Utils.noise(graveValue, params.epsilon, this.m_rnd.nextDouble());

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (graveValue > bestValue) {
                selected = child;
                bestValue = graveValue;
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        roll(state, actions[selected.childIdx]);

        return selected;
    }

    private SingleTreeNode cutOFFAMAF(GameState state ,int numInterations) {
        int k = params.cutOffValue;
        System.out.println(k);
        System.out.println(numInterations);
        SingleTreeNode selected = null;
        if (numInterations < k)
        {
            selected = AMAF(state);
        }
        else
        {
            selected = uct(state);
        }
        return selected;
    }


    private double rollOut(GameState state)
    {
        int thisDepth = this.m_depth;

        while (!finishRollout(state,thisDepth)) {
            int action = safeRandomAction(state);
            roll(state, actions[action]);
            thisDepth++;
        }

        return rootStateHeuristic.evaluateState(state);
    }

    private int safeRandomAction(GameState state)
    {
        Types.TILETYPE[][] board = state.getBoard();
        ArrayList<Types.ACTIONS> actionsToTry = Types.ACTIONS.all();
        int width = board.length;
        int height = board[0].length;

        while(actionsToTry.size() > 0) {

            int nAction = m_rnd.nextInt(actionsToTry.size());
            Types.ACTIONS act = actionsToTry.get(nAction);
            Vector2d dir = act.getDirection().toVec();

            Vector2d pos = state.getPosition();
            int x = pos.x + dir.x;
            int y = pos.y + dir.y;

            if (x >= 0 && x < width && y >= 0 && y < height)
                if(board[y][x] != Types.TILETYPE.FLAMES)
                    return nAction;

            actionsToTry.remove(nAction);
        }

        //Uh oh...
        return m_rnd.nextInt(num_actions);
    }

    @SuppressWarnings("RedundantIfStatement")
    private boolean finishRollout(GameState rollerState, int depth)
    {
        if (depth >= params.rollout_depth)      //rollout end condition.
            return true;

        if (rollerState.isTerminal())               //end of game
            return true;

        return false;
    }

    private void backUp(SingleTreeNode node, double result)
    {
        SingleTreeNode n = node;
        while(n != null)
        {
            n.nVisits++;
            n.totValue += result;
            n.samples.add(result);
            if (result < n.bounds[0]) {
                n.bounds[0] = result;
            }
            if (result > n.bounds[1]) {
                n.bounds[1] = result;
            }
            n = n.parent;
        }
    }
    private void backUpAMAF(SingleTreeNode node, double result)
    {
        SingleTreeNode n = node;
        SingleTreeNode n_parent;
        while(n != null)
        {
            n.nVisits++;
            n.nAMAFVisits++;
            n.totAMAFValue = n.totAMAFValue + result;
            n.totValue = n.totValue + result;
            n.samples.add(result);
            if (result < n.bounds[0] && result < n.AMAFbounds[0]) {
                n.bounds[0] = result;
                if (n.AMAFbounds[0] ==  Double.MAX_VALUE || n.AMAFbounds[0] == -Double.MAX_VALUE)
                    n.AMAFbounds[0] = result;
                else
                    n.AMAFbounds[0] = n.AMAFbounds[0]+result;
            }
            if (result > n.bounds[1] && result > AMAFbounds[1]) {
                n.bounds[1] = result;
                if (n.AMAFbounds[1] ==  Double.MAX_VALUE || n.AMAFbounds[1] == -Double.MAX_VALUE)
                    n.AMAFbounds[1] = result;
                else
                    n.AMAFbounds[1] = n.AMAFbounds[1]+result;
            }

            n_parent = n.parent;
            while (n_parent != null){
                for(SingleTreeNode child: n_parent.children){
                    if (child == n && child != node){
                        child.nAMAFVisits++;
                        child.totAMAFValue = child.totAMAFValue + result;
                        if (result < child.AMAFbounds[0]) {
                            if (child.AMAFbounds[0] ==  Double.MAX_VALUE || child.AMAFbounds[0] == -Double.MAX_VALUE)
                                child.AMAFbounds[0] = result;
                            else
                                child.AMAFbounds[0] = child.AMAFbounds[0]+result;
                        }
                        if (result > child.AMAFbounds[1]) {
                            if (child.AMAFbounds[1] ==  Double.MAX_VALUE || child.AMAFbounds[1] == -Double.MAX_VALUE)
                                child.AMAFbounds[1] = result;
                            else
                                child.AMAFbounds[1] = child.AMAFbounds[1]+result;
                        }
                    }
                }
                n_parent = n_parent.parent;
            }
            n = n.parent;
        }
    }

    int mostVisitedAction() {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;
        boolean allEqual = true;
        double first = -1;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null)
            {
                if(first == -1)
                    first = children[i].nVisits;
                else if(first != children[i].nVisits)
                {
                    allEqual = false;
                }

                double childValue = children[i].nVisits;
                childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            selected = 0;
        }else if(allEqual)
        {
            //If all are equal, we opt to choose for the one with the best Q.
            selected = bestAction();
        }

        return selected;
    }

    private int bestAction()
    {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null) {
                double childValue = children[i].totValue / (children[i].nVisits + params.epsilon);
                childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }

        return selected;
    }


    private boolean notFullyExpanded() {
        for (SingleTreeNode tn : children) {
            if (tn == null) {
                return true;
            }
        }

        return false;
    }
}
