"""Replace Chapter 2 with _ch2_new.html, correct references, add references 43-55 and the verification note. Idempotent."""
import re
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "report.html"
s = open(p, encoding="utf-8").read()
if 'id="s2-11"' in s:
    print("already patched")
    sys.exit(0)
new = open("_ch2_new.html", encoding="utf-8").read().strip()
m = re.search(r'<section class="sheet" id="ch2">.*?</section>', s, flags=re.S)
assert m
s = s[:m.start()] + new + s[m.end():]


def sub_li(prefix, newtext):
    global s
    m = re.search(r"<li>" + re.escape(prefix) + r".*?</li>", s, flags=re.S)
    assert m, prefix
    s = s[:m.start()] + "<li>" + newtext + "</li>" + s[m.end():]


sub_li("Haarnoja, T. et al. (2018). Soft Actor-Critic.", "Haarnoja, T. et al. (2018). Soft Actor-Critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor. <em>ICML 2018</em>, arXiv:1801.01290.")
sub_li("Sundaralingam, B.", "Sundaralingam, B. et al. (2023). cuRobo: parallelized collision-free minimum-jerk robot motion generation. arXiv:2310.17274.")
sub_li("Qureshi, A. H. et al. (2019). Motion Planning Networks.", "Qureshi, A. H. et al. (2019). Motion Planning Networks. <em>ICRA 2019</em>, 2118&ndash;2124, arXiv:1806.05767.")
sub_li("Deep-reinforcement-learning-based path planning", "Bhuiyan, T. et al. (2023). Deep-reinforcement-learning-based path planning for industrial robots using distance sensors as observation. arXiv:2301.05980.")
sub_li("URPlanner:", "Ying, F. et al. (2025). URPlanner: a universal paradigm for collision-free robotic motion planning based on deep reinforcement learning. arXiv:2505.20175.")
sub_li("Agarwal, R. et al. (2021).", "Agarwal, R. et al. (2021). Deep reinforcement learning at the edge of the statistical precipice. <em>NeurIPS 2021</em>, arXiv:2108.13264.")
sub_li("Zeng, A. et al. (2020).", "Zeng, A. et al. (2020). TossingBot: learning to throw arbitrary objects with residual physics. <em>IEEE Transactions on Robotics</em> 36(4), 1307&ndash;1319, arXiv:1903.11239.")
sub_li("Ibarz, J. et al. (2021).", "Ibarz, J. et al. (2021). How to train your robot with deep reinforcement learning: lessons we have learned. <em>The International Journal of Robotics Research</em> 40(4&ndash;5), 698&ndash;721.")
sub_li("Cheng, R.", "Cheng, R., Orosz, G., Murray, R. M., Burdick, J. W. (2019). End-to-end safe reinforcement learning through barrier functions for safety-critical continuous control tasks. <em>AAAI 2019</em>, 33(1), 3387&ndash;3395, arXiv:1903.08792.")
sub_li("Alshiekh, M.", "Alshiekh, M. et al. (2018). Safe reinforcement learning via shielding. <em>AAAI 2018</em>, 32(1), arXiv:1708.08611.")
sub_li("Ames, A. D.", "Ames, A. D. et al. (2019). Control barrier functions: theory and applications. <em>European Control Conference 2019</em>, 3420&ndash;3431, arXiv:1903.11199.")
sub_li("Tobin, J.", "Tobin, J. et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. <em>IROS 2017</em>, 23&ndash;30, arXiv:1703.06907.")
sub_li("Todorov, E.", "Todorov, E., Erez, T., Tassa, Y. (2012). MuJoCo: a physics engine for model-based control. <em>IROS 2012</em>, 5026&ndash;5033.")
sub_li("Elephant Robotics. myCobot 280 user documentation", "Elephant Robotics. myCobot documentation (GitBook). https://docs.elephantrobotics.com/docs/gitbook-en/ (accessed 6 November 2026).")
sub_li("Elephant Robotics. <code>pymycobot</code>", "Elephant Robotics. <code>pymycobot</code>, Python API for the myCobot arms. https://github.com/elephantrobotics/pymycobot (accessed 6 November 2026).")
sub_li("Elephant Robotics. <code>mycobot_ros</code>", "Elephant Robotics. <code>mycobot_ros</code>, ROS packages and URDF and mesh description of the myCobot arms. https://github.com/elephantrobotics/mycobot_ros (accessed 6 November 2026).")

more = [
    "Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. <em>Nature</em> 518(7540), 529&ndash;533.",
    "Schaul, T. et al. (2015). Universal value function approximators. <em>ICML 2015</em>, PMLR 37.",
    "Plappert, M. et al. (2018). Multi-goal reinforcement learning: challenging robotics environments and request for research. arXiv:1802.09464.",
    "Levine, S. et al. (2016). End-to-end training of deep visuomotor policies. <em>Journal of Machine Learning Research</em> 17(39), 1&ndash;40.",
    "Kalashnikov, D. et al. (2018). QT-Opt: scalable deep reinforcement learning for vision-based robotic manipulation. <em>CoRL 2018</em>, PMLR 87, arXiv:1806.10293.",
    "Gu, S. et al. (2017). Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. <em>ICRA 2017</em>, 3389&ndash;3396, arXiv:1610.00633.",
    "Peng, X. B. et al. (2018). Sim-to-real transfer of robotic control with dynamics randomization. <em>ICRA 2018</em>, 3803&ndash;3810, arXiv:1710.06537.",
    "Chebotar, Y. et al. (2019). Closing the sim-to-real loop: adapting simulation randomization with real world experience. <em>ICRA 2019</em>, 8973&ndash;8979, arXiv:1810.05687.",
    "Zhao, T. Z. et al. (2023). Learning fine-grained bimanual manipulation with low-cost hardware. <em>Robotics: Science and Systems XIX</em>, arXiv:2304.13705.",
    "Brunke, L. et al. (2022). Safe learning in robotics: from learning-based control to safe reinforcement learning. <em>Annual Review of Control, Robotics, and Autonomous Systems</em> 5, 411&ndash;444, arXiv:2108.06266.",
    "Qureshi, A. H. et al. (2021). Motion planning networks: bridging the gap between learning-based and classical motion planners. <em>IEEE Transactions on Robotics</em> 37(1), 48&ndash;66.",
    "Kawaharazuka, K. et al. (2022). Self-supervised learning of visual servoing for low-rigidity robots considering temporal body changes. <em>IEEE Robotics and Automation Letters</em> 7(3), 7881&ndash;7887.",
    "Kawaharazuka, K. et al. (2023). Daily assistive view control learning of low-cost low-rigidity robot via large-scale vision-language model. <em>IEEE-RAS Humanoids 2023</em>, 1&ndash;6, arXiv:2312.07451.",
]
items = "\n".join("    <li>%s</li>" % r for r in more)
m = re.search(r"(<li>Elephant Robotics\. <code>mycobot_ros</code>.*?</li>)", s, flags=re.S)
s = s[:m.end()] + "\n" + items + s[m.end():]
m = re.search(r'  <div class="todo-block"><strong class="tag">To verify</strong>References 16.*?</div>\n', s, flags=re.S)
assert m
note = ('  <p style="font-family:var(--font-ui);font-size:13.5px;color:var(--muted)"><strong>Verification.</strong> On 6 November 2026 every arXiv number was checked '
        'against the arXiv listing (title, first author, year), and the journal and conference entries against Crossref (title, first author, venue, year, volume, pages) '
        'or the publisher&rsquo;s page; three titles and two author lists were corrected (references 4, 8, 11 and 12). Not verified in full: reference 17 is a technical '
        'report whose number appears as 98-11 and as 9811 in different catalogues; the proceedings volumes cited as PMLR (references 44 and 47) are given without page numbers; '
        'references 40&ndash;42 are web pages whose existence and titles were checked but whose content was not; the arXiv-only entries (2, 6, 29, 33, 39, 45) are preprints. '
        'Co-authors are given as &ldquo;et al.&rdquo; wherever the full list was not checked.</p>\n')
s = s[:m.start()] + note + s[m.end():]
open(p, "w", encoding="utf-8").write(s)
print("patched chapter 2 and references")
