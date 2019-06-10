from base_trainer.net_work import trainner
import setproctitle

setproctitle.setproctitle("detect")

trainner=trainner()

trainner.train()
