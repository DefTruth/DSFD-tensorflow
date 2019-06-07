from base_trainer.net_work import trainner
import setproctitle

setproctitle.setproctitle("face-like")

trainner=trainner()

trainner.train()
