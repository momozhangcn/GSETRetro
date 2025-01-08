#!/usr/bin/env python
from onmt.bin.average_models import main


if __name__ == "__main__":
    main()




'''/home/zhangmeng/aMy-ONMT/
python average_models.py -models experiments/biochem_npl_10xaug_RSmiles_6x512/model_step_375000.pt \
            experiments/biochem_npl_10xaug_RSmiles_6x512/model_step_375000.pt \
            -output  experiments/biochem_npl_10xaug_RSmiles_6x512/model_step_avg_375000.pt


/home/zhangmeng/aMy-ONMT003/experiments/biochem_npl_10xaug_RSmiles_6x512/model_step_avg_375000.pt
 /home/zhangmeng/ALL-tied2way-transformer/tied-twoway-transformer-2B4Biochem+NPL/onmt-runs/model/model_step_100000.pt  /home/zhangmeng/ALL-tied2way-transformer/tied-twoway-transformer-2B4Biochem+NPL/onmt-runs/model/model_step_120000.pt  /home/zhangmeng/ALL-tied2way-transformer/tied-twoway-transformer-3C4justNPL/onmt-runs/model/model_step_100000.pt    /home/zhangmeng/ALL-tied2way-transformer/tied-twoway-transformer-3C4justNPL/onmt-runs/model/model_step_120000.pt
'''
'''

'''