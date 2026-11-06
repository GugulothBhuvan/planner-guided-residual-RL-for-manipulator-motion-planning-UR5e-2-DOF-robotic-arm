"""Report rewrite, part 1: MyCobot framing (cover line, abstract, Chapter 1, wording, Chapter 6 opener, Chapter 7). Idempotent."""
import re
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "report.html"
s = open(p, encoding="utf-8").read()
if "Why develop on other arms first" in s:
    print("already patched")
    sys.exit(0)


def rep(old, new, must=True):
    global s
    if old not in s:
        assert not must, old[:90]
        return
    s = s.replace(old, new, 1)


def rep_re(pat, new, must=True):
    global s
    m = re.search(pat, s, flags=re.S)
    if not m:
        assert not must, pat[:90]
        return
    s = s[:m.start()] + (new(m) if callable(new) else new) + s[m.end():]


rep("A UR5e study with a planar 2-DOF companion and a real-arm test on the MyCobot 280",
    "Developed on a simulated UR5e and a planar 2-DOF arm, deployed on a MyCobot 280")

abstract = (
    "The target of this work is a MyCobot 280, a small low-cost six-joint arm that must reach tool poses among obstacles through a 20 Hz command loop "
    "limited to 0.5&deg; per step. A cheap sampling-based planner is attractive on such a robot, but its success falls as its collision-check budget shrinks. "
    "The method keeps a cheap RRT-Connect planner (1,500 to 6,000 collision checks) and a tracker that follows its path, and adds a Soft Actor-Critic residual "
    "to the tracker&rsquo;s command, in two designs: an additive correction and an accumulated joint offset. The method was developed on a simulated UR5e and a "
    "planar 2-DOF arm, which offer a tested public model and an exactly computable configuration space, because the MyCobot first needed its own measurements. "
    "On 200 held-out UR5e scenes, full RRT-Connect succeeds in 99.0% of episodes, the cheap planner alone in 79.5&ndash;90.0%, and the additive residual lowers "
    "success to 66&ndash;75%; the offset residual raises collisions from 8.5&ndash;20.5% to about 40%. I call this the authority&ndash;safety dilemma: a correction "
    "small enough to stay on the planner&rsquo;s safe path cannot fix its failures, and one large enough to fix them leaves the safe path. The planar arm reproduces "
    "the pattern and agrees with the UR5e on 8 of 9 design comparisons. On the MyCobot in simulation the retrained residual again matches its planner (93.0&ndash;93.5% "
    "against 93.5% at 6,000 checks), also on harder challenge sets and with box and cylinder obstacles. On the real arm, 30 streamed runs with virtual obstacles all "
    "reached the goal within 1 cm and 5&deg; once a small final-approach correction was added, and the residual was indistinguishable from its planner. Streaming "
    "limits the arm to 10&deg;/s, so the same paths were also sent as firmware segments: 30 of 30 runs again, with a median time of 5&ndash;6 s instead of 8&ndash;10 s. "
    "In segment mode the residual is replayed from simulation, so only the streamed runs test the policy on hardware."
)
rep_re(r'(<section[^>]*id="abstract".*?</h1>\s*)<p>.*?</p>', lambda m: m.group(1) + "<p>" + abstract + "</p>")

rep("<p>A cheap planner plus a learned residual would give a cheap, safe and robust controller if the correction fixed the planner&rsquo;s failures and left its successes alone. Three things were unknown at the start of the project:</p>",
    "<p>The target robot of this thesis is the MyCobot 280, a small, low-cost, position-controlled arm with a servo deadband and a slow command loop, where the cost of planning and the safety of the motion both matter. A cheap planner plus a learned residual would give a cheap, safe and robust controller on such an arm if the correction fixed the planner&rsquo;s failures and left its successes alone. Three things were unknown at the start of the project:</p>")
rep("<li><strong>Whether the answer holds on a real arm.</strong>", "<li><strong>Whether the answer holds on the target arm, in simulation and on hardware.</strong>")
chain = '''  <h3>Why develop on other arms first</h3>
  <p>The method was developed on two other arms before it was deployed on the MyCobot, for three reasons (Figure 1.1). The UR5e is a standard industrial arm with a well-tested public model (MuJoCo Menagerie), so errors in the environment, the planner and the training code could be separated from errors in the robot model. The planar 2-DOF arm has an exactly computable configuration space, so design questions could be checked cheaply and against the true answer. The MyCobot is small and slow and had to be measured (step size, latency, servo deadband, safe joint limits) before it could be simulated faithfully. The UR5e residual was then run on the MyCobot model without retraining, the residual was retrained on the MyCobot, and the result was taken to the real arm, first by streaming a target every 50 ms and then by sending the paths as firmware segments.</p>
  <figure id="fig-1-1">
    <div class="frame"><img src="img/deployment_chain.png" alt="Deployment chain: develop on the UR5e and the planar arm, zero-shot probe on the MyCobot model, retrain on the MyCobot, real arm by streaming and by firmware segments" loading="lazy" style="max-width:46rem"></div>
    <figcaption><b>Figure 1.1:</b> The deployment chain. The method was developed on the UR5e and the planar 2-DOF arm, probed on the MyCobot model without retraining, retrained on it, and run on the real MyCobot 280 by streaming (the policy runs on the hardware) and by firmware segments (the residual is replayed from simulation).</figcaption>
  </figure>
'''
rep('  <h2 id="s1-6">', chain + '  <h2 id="s1-6">')
rep("Let <var>q</var><sub>0</sub> be the start configuration of an arm and <var>g</var> a target tool pose, with a set of obstacles in the workspace.",
    "Let <var>q</var><sub>0</sub> be the start configuration of an arm (the target is the MyCobot 280) and <var>g</var> a target tool pose, with a set of obstacles in the workspace.")
rep("<li>to check whether the findings hold on a different arm and on real hardware.</li>",
    "<li>to deploy the controller on the target arm, the MyCobot 280, in simulation and on the real arm, and to check whether the findings from the development arms hold there.</li>")
rep("Do the findings hold on a different arm, on harder scenes and with non-spherical obstacles, and on real hardware?",
    "Do the findings hold on the target arm (the MyCobot 280), on harder scenes and with non-spherical obstacles, and on real hardware?")
rep("MyCobot lane: simulation, challenge sets, shape set, real-arm runs", "Target-robot results: MyCobot 280 simulation, challenge sets, shape set, real-arm runs")

for a, b in [
    (" In a second lane, the pipeline was ported to the MyCobot 280.", ""),
    ("<strong>A real-arm deployment.</strong> A MyCobot 280 lane with its own", "<strong>A target-robot deployment.</strong> A MyCobot 280 deployment with its own"),
    ("the UR5e study, the planar study and the MyCobot lane", "the UR5e study, the planar study and the target-robot results"),
    ("differ between the UR5e and the MyCobot lane", "differ between the UR5e and the MyCobot 280"),
    ("4.6&nbsp;&nbsp;Port to the MyCobot 280", "4.6&nbsp;&nbsp;Deployment on the MyCobot 280"),
    ("The MyCobot lane applies", "The MyCobot deployment applies"),
    ("between the UR5e study and the MyCobot lane.", "between the UR5e study and the MyCobot 280 deployment."),
    ("<td>MyCobot port</td>", "<td>MyCobot deployment</td>"),
    ("The same structure was ported to a real arm", "The same structure was carried over to a real arm"),
    ("v0.7 for the MyCobot lane", "v0.7 for the MyCobot deployment"),
    ("The MyCobot lane uses the same environment code", "The MyCobot deployment uses the same environment code"),
    ("The MyCobot lane builds its own S4 sets", "The MyCobot deployment builds its own S4 sets"),
    ("the planar package 74 and the MyCobot lane 11", "the planar package 74 and the MyCobot package 11"),
    ("Refinements on the MyCobot lane", "Refinements on the MyCobot 280"),
    ("The MyCobot lane has its own scenes", "The MyCobot 280 has its own scenes"),
    ("several lane-specific refinements", "several MyCobot-specific refinements"),
    ("The lane first asked whether", "The deployment first asked whether"),
    ("The MyCobot lane shows that the finding survives", "The MyCobot deployment shows that the finding survives"),
    ("<h3>MyCobot lane (from <code>mycobot/</code>)</h3>", "<h3>MyCobot 280 (from <code>mycobot/</code>)</h3>"),
]:
    rep(a, b, must=False)

rep_re(r"This chapter reports the results in the order of the research questions\. Sections 6\.2 to 6\.4 cover.*?Section 6\.10 answers the research questions\.",
       "This chapter reports the results in the order of the research questions. Sections 6.2 to 6.5 are development results: they were obtained on the UR5e (reaching without obstacles, motion planning in clutter, and the diagnosis of why the residual fails) and on the planar 2-DOF arm, to find out what the method does and why. Sections 6.6 to 6.9 are target-robot results: the MyCobot 280 in simulation, on the challenge and shape sets, and on the real arm. Section 6.10 answers the research questions, starting from what holds on the MyCobot.")
rep("<strong>No.</strong> The additive residual is below its planner at every budget; the offset residual raises collisions two to five times. On the MyCobot the residual equals its planner.",
    "<strong>No.</strong> On the MyCobot, the target arm, the residual equals its planner. On the development arms the additive residual is below its planner at every budget and the offset residual raises collisions two to five times.")
rep("The residual equals its planner on a second arm, on the real arm, on harder scenes and on box and cylinder obstacles.",
    "On the target arm the residual equals its planner in simulation, on the real arm (streamed runs), on harder scenes and on box and cylinder obstacles, as it did on the development arms.")

m = re.search(r'(<p><strong>RQ6\.</strong>.*?</p>\s*)', s, flags=re.S)
if m:
    para = m.group(1).replace("<strong>RQ6.</strong>", "<strong>Target robot (RQ6).</strong>", 1)
    s = s[:m.start()] + s[m.end():]
    k = s.find("<p><strong>RQ1 and RQ2.</strong>")
    assert k > 0
    s = s[:k] + para + s[k:]
rep("A MuJoCo benchmark for the UR5e, two residual designs, a planar companion study and a deployment on a real MyCobot 280 were built to answer it.",
    "The target was the MyCobot 280; a MuJoCo benchmark for the UR5e, two residual designs and a planar 2-DOF study were built to develop the method, and it was then deployed on the MyCobot 280, in simulation and on the real arm.")
open(p, "w", encoding="utf-8").write(s)
print("patched framing")
