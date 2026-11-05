"""Report patch: related work (Sections 2.2, new 2.7, Table 2.1 rows) and references 16-42. Run once; idempotent."""
import re
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "report.html"
s = open(p, encoding="utf-8").read()
if "id=\"s2-7-safe\"" in s:
    print("already patched")
    sys.exit(0)


def rep(old, new, count=1):
    global s
    assert old in s, old[:100]
    s = s.replace(old, new, count)


def rep_re(pat, new):
    global s
    m = re.search(pat, s, flags=re.S)
    assert m, pat[:100]
    s = s[:m.start()] + new + s[m.end():]


# ---- chapter introduction -----------------------------------------------------------------------------------
rep("Section 2.6 residual policy learning, which is closest to this work. Section 2.7 notes the evaluation conventions followed.",
    "Section 2.6 residual policy learning, which is closest to this work. Section 2.7 covers safe learning and the transfer to real robots, "
    "and Section 2.8 notes the evaluation conventions followed.")

# ---- 2.2 ------------------------------------------------------------------------------------------------------
rep_re(r'  <div class="todo-block"><strong class="tag">To add</strong>Other planning families.*?</div>\n',
       '''  <p>Rapidly exploring random trees were introduced as a way to search high-dimensional spaces by random growth [17]. Probabilistic roadmaps take the other classical route: they sample a graph of collision-free configurations once and answer many queries in the same scene from it [16]. Asymptotically optimal variants (RRT*, PRM*) add rewiring so that the path improves with more samples, at the price of more computation [18]. Planners of this kind are packaged in the libraries normally used with manipulators: the Open Motion Planning Library [19] and MoveIt, which builds on it [20]. A second family treats planning as trajectory optimisation. CHOMP improves an initial trajectory by covariant gradient descent on a smoothness and obstacle cost [21]; STOMP perturbs the trajectory with noise and needs no gradient of the cost [22]; TrajOpt solves a sequence of convex problems with convex collision checking [23]. Optimisation gives smooth trajectories but depends on the initial guess and can stop in a local minimum, so it is often run after, or beside, a sampling-based planner. This report keeps RRT-Connect because its cost is a count of collision checks that can be capped, and that cap is the budget the study varies.</p>
''')

# ---- new section 2.7, renumber 2.7 -> 2.8, 2.8 -> 2.9, 2.9 -> 2.10 --------------------------------------------
rep('<h2 id="s2-9">2.9&nbsp;&nbsp;Chapter Summary</h2>', '<h2 id="s2-10">2.10&nbsp;&nbsp;Chapter Summary</h2>')
rep('<h2 id="s2-8">2.8&nbsp;&nbsp;Limitations of Existing Methods and Research Gap</h2>', '<h2 id="s2-9">2.9&nbsp;&nbsp;Limitations of Existing Methods and Research Gap</h2>')
rep('<h2 id="s2-7">2.7&nbsp;&nbsp;Evaluation Conventions</h2>', '''<h2 id="s2-7-safe">2.7&nbsp;&nbsp;Safe Learning and Transfer to Real Robots</h2>
  <p>Combining a learned component with a planner is attractive because the planner can supply the guarantee that learning lacks, and several lines of work use this idea. Learning a sampling distribution for a sampling-based planner keeps the planner and its collision checking and uses the network only to propose where to sample [24]. PRM-RL builds a roadmap whose edges are accepted only if a learned point-to-point policy can follow them, so the learned part acts as the local planner; it was shown on mobile robots [25]. This report differs in that the learned part corrects the commands that track a finished path.</p>
  <p>The authority&ndash;safety dilemma found in Chapter 6 is the problem that safe reinforcement learning addresses. Surveys separate methods that change the objective, for example constrained policy optimisation [27], from methods that change the action [26]. A shield replaces actions that would leave the safe set by safe ones [28]; a safety layer corrects the action by a linear model of the constraint that is learned from data [29]; and a control barrier function gives a condition on the action that keeps the state inside a safe set when it holds at every step [30]. Cheng et al. combine a barrier-function controller with a model-free policy and learn the model error, so that exploration stays safe [31]. These are the ideas behind the safety-filtered residual of Section 7.5: a filter of this kind would let the residual be free enough to correct the planner and still stop it from leaving the planner&rsquo;s safe path.</p>
  <p>Learning for a physical arm adds the gap between simulation and the real arm. Domain randomisation trains over varied simulator parameters so that the real arm looks like one more variation [34]; surveys of simulation-to-real transfer [35] and a review of lessons from training real robots with reinforcement learning [36] list the other remedies. Residual methods are one of them, because the analytic part carries the model and only the error is learned: TossingBot learns a residual on top of a physics-based controller [32], and residual reinforcement learning from demonstrations speeds up learning on sparse-reward manipulation tasks [33]. In this study the policy is trained in MuJoCo [37] with Stable-Baselines3 [38] and run on the MyCobot 280 [40&ndash;42]. Dynamics were not randomised; the servo deadband of the real arm is not modelled (Section 4.6), and the hardware runs are evaluated against that.</p>

  <h2 id="s2-8-eval">2.8&nbsp;&nbsp;Evaluation Conventions</h2>''')
# the original 2.7 heading is now followed by a duplicated block-title; keep the evaluation paragraph under 2.8 (done above).

# ---- Table 2.1: two more rows -------------------------------------------------------------------------------
rep("<tr><td>Motion Planning Networks, Motion Policy Networks [9, 10]</td>",
    "<tr><td>CHOMP, STOMP, TrajOpt (optimisation) [21&ndash;23]</td><td>No (optimises a trajectory)</td><td>No</td><td>n/a</td><td>Yes</td></tr>\n    "
    "<tr><td>Motion Planning Networks, Motion Policy Networks [9, 10]</td>")
rep("<tr><td>Residual policy learning [6, 7]</td>",
    "<tr><td>PRM-RL [25]</td><td>Yes (roadmap)</td><td>Learns the local policy</td><td>No</td><td>Mobile robots</td></tr>\n    "
    "<tr><td>Residual policy learning [6, 7]</td>")

# ---- chapter summary --------------------------------------------------------------------------------------------
rep_re(r'  <div class="todo-block"><strong class="tag">To expand</strong>The literature review rests on 15 references.*?</div>\n', '')
rep("Classical planners are reliable with a large budget and fragile with a small one. Learned planners replace them, and residual learning corrects reactive controllers. Chapter 3 formulates the problem that combines the two.",
    "Classical planners are reliable with a large budget and fragile with a small one, and optimisation-based planners depend on their starting trajectory. Learned planners replace the classical ones or guide them, and residual learning corrects reactive controllers. Safe reinforcement learning offers filters that keep a learned correction inside a safe set, and domain randomisation and residual methods are the usual remedies for the gap to a real arm. None of these studies a residual on top of a base that tracks a planned path. Chapter 3 formulates the problem that combines the two.")

# ---- references ------------------------------------------------------------------------------------------------------
new_refs = [
    "Kavraki, L. E., Svestka, P., Latombe, J.-C., Overmars, M. H. (1996). Probabilistic roadmaps for path planning in high-dimensional configuration spaces. <em>IEEE Transactions on Robotics and Automation</em> 12(4), 566&ndash;580.",
    "LaValle, S. M. (1998). Rapidly-exploring random trees: a new tool for path planning. Technical Report 98-11, Iowa State University.",
    "Karaman, S., Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning. <em>The International Journal of Robotics Research</em> 30(7), 846&ndash;894.",
    "Sucan, I. A., Moll, M., Kavraki, L. E. (2012). The Open Motion Planning Library. <em>IEEE Robotics &amp; Automation Magazine</em> 19(4), 72&ndash;82.",
    "Chitta, S., Sucan, I., Cousins, S. (2012). MoveIt! <em>IEEE Robotics &amp; Automation Magazine</em> 19(1), 18&ndash;19.",
    "Ratliff, N., Zucker, M., Bagnell, J. A., Srinivasa, S. (2009). CHOMP: gradient optimization techniques for efficient motion planning. <em>ICRA 2009</em>.",
    "Kalakrishnan, M., Chitta, S., Theodorou, E., Pastor, P., Schaal, S. (2011). STOMP: stochastic trajectory optimization for motion planning. <em>ICRA 2011</em>.",
    "Schulman, J. et al. (2014). Motion planning with sequential convex optimization and convex collision checking. <em>The International Journal of Robotics Research</em> 33(9), 1251&ndash;1270.",
    "Ichter, B., Harrison, J., Pavone, M. (2018). Learning sampling distributions for robot motion planning. <em>ICRA 2018</em>, arXiv:1709.05448.",
    "Faust, A. et al. (2018). PRM-RL: long-range robotic navigation tasks by combining reinforcement learning and sampling-based planning. <em>ICRA 2018</em>, arXiv:1710.03937.",
    "Garc&iacute;a, J., Fern&aacute;ndez, F. (2015). A comprehensive survey on safe reinforcement learning. <em>Journal of Machine Learning Research</em> 16, 1437&ndash;1480.",
    "Achiam, J., Held, D., Tamar, A., Abbeel, P. (2017). Constrained policy optimization. <em>ICML 2017</em>, arXiv:1705.10528.",
    "Alshiekh, M. et al. (2018). Safe reinforcement learning via shielding. <em>AAAI 2018</em>, arXiv:1708.08611.",
    "Dalal, G. et al. (2018). Safe exploration in continuous action spaces. arXiv:1801.08757.",
    "Ames, A. D. et al. (2019). Control barrier functions: theory and applications. <em>European Control Conference 2019</em>, arXiv:1903.11199.",
    "Cheng, R., Orosz, G., Murray, R. M., Burdick, J. W. (2019). End-to-end safe reinforcement learning through barrier functions for safety-critical continuous control tasks. <em>AAAI 2019</em>, arXiv:1903.08792.",
    "Zeng, A. et al. (2020). TossingBot: learning to throw arbitrary objects with residual physics. <em>IEEE Transactions on Robotics</em> 36(4), arXiv:1903.11239.",
    "Alakuijala, M. et al. (2021). Residual reinforcement learning from demonstrations. arXiv:2106.08050.",
    "Tobin, J. et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. <em>IROS 2017</em>, arXiv:1703.06907.",
    "Zhao, W., Queralta, J. P., Westerlund, T. (2020). Sim-to-real transfer in deep reinforcement learning for robotics: a survey. <em>IEEE SSCI 2020</em>, arXiv:2009.13303.",
    "Ibarz, J. et al. (2021). How to train your robot with deep reinforcement learning: lessons we have learned. <em>The International Journal of Robotics Research</em> 40(4&ndash;5).",
    "Todorov, E., Erez, T., Tassa, Y. (2012). MuJoCo: a physics engine for model-based control. <em>IROS 2012</em>.",
    "Raffin, A. et al. (2021). Stable-Baselines3: reliable reinforcement learning implementations. <em>Journal of Machine Learning Research</em> 22(268), 1&ndash;8.",
    "Haarnoja, T. et al. (2018). Soft Actor-Critic algorithms and applications. arXiv:1812.05905.",
    "Elephant Robotics. myCobot 280 user documentation and product manual. https://docs.elephantrobotics.com",
    "Elephant Robotics. <code>pymycobot</code>, Python API for the myCobot arms. https://github.com/elephantrobotics/pymycobot",
    "Elephant Robotics. <code>mycobot_ros</code>, ROS packages and URDF and mesh description of the myCobot arms. https://github.com/elephantrobotics/mycobot_ros",
]
items = "\n".join(f"    <li>{r}</li>" for r in new_refs)
rep("<li>Agarwal, R. et al. (2021). Deep RL at the edge of the statistical precipice. arXiv:2108.13264.</li>",
    "<li>Agarwal, R. et al. (2021). Deep RL at the edge of the statistical precipice. arXiv:2108.13264.</li>\n" + items)
rep_re(r'  <div class="todo-block"><strong class="tag">To add</strong>The reference list has 15 entries.*?</div>\n',
       '  <div class="todo-block"><strong class="tag">To verify</strong>References 16&ndash;42 were added from the standard literature and from the manufacturer&rsquo;s public pages. Check the venues, page numbers and web addresses against the originals before submission; web pages need an access date.</div>\n')
# any later mention of the removed counts
open(p, "w", encoding="utf-8").write(s)
print("patched refs")
