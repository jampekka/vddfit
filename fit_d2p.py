from itertools import groupby
import pandas as pd
import numpy as np
from vdd_disc import vdd_loss, vdd_decision_pdf, VddmParams, vdd_blocker_loss, vdd_blocker_decision_pdf
import tdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize
from pprint import pprint
import hikersim
from crossmods import Vddm, LognormalTdm as Tdm, Grid1d
from collections import defaultdict
from kwopt import minimizer, logbarrier, logitbarrier, fixed
import kwopt
#_minimizer = minimizer
#def minimizer(loss, **kwargs):
#    return _minimizer(loss, opt_f=scipy.optimize.basinhopping, minimizer_kwargs=kwargs)


DT = 1/30
vddm_params = {
    #'keio_uk': {'std': 0.6895813463269477, 'damping': 1.465877152701145, 'scale': 0.6880628317112305, 'tau_threshold': 2.267405295940994, 'act_threshold': 0.976580507453147, 'pass_threshold': -0.0077883119117825046},
    #'keio_uk': {'std': 1.0, 'damping': 1.0, 'scale': 1.0, 'tau_threshold': 2.0, 'act_threshold': 1.0, 'pass_threshold': 0.0, 'dot_coeff': 1e-6, 'ehmi_coeff': 0.0, 'dist_coeff': 0.0},
    'keio_japan': {'std': 0.6200239180599325, 'damping': 1.7363852589095383, 'scale': 0.34685222829243945, 'tau_threshold': 2.179287004380974, 'act_threshold': 0.9509308457583107, 'pass_threshold': -0.016836928176652676},
    #'hiker': {'std': 0.5229662639983371, 'damping': 0.9946376439854723, 'scale': 0.48663037841127904, 'tau_threshold': 2.308618665956897, 'act_threshold': 1.053645908407938, 'pass_threshold': 1.024761904761648},
    #'hiker': {'std': 0.5935959086407167, 'damping': 0.9151503732988887, 'scale': 0.39910022137223705, 'tau_threshold': 2.533825817019247, 'act_threshold': 1.0588392695282198, 'pass_threshold': 0.9900001990325101, 'dot_coeff': 1.1290275837238286, 'ehmi_coeff': 0.33449603099173664},
    #'hiker': {'std': 0.6242299905405555, 'damping': 0.8077397267051836, 'scale': 0.37777720233236123, 'tau_threshold': 3.0037091915789738, 'act_threshold': 1.0, 'pass_threshold': 0.9568042787179346, 'dot_coeff': 1.517345748330563, 'ehmi_coeff': 0.4127020530271129, 'dist_coeff': 0.29191574142779203}
    #'hiker': {'std': 0.6737559893535666, 'damping': 0.7602189090471121, 'scale': 0.4657452690259603, 'tau_threshold': 3.158682267433862, 'act_threshold': 1.1100000438010933, 'pass_threshold': 0.9871867080244127, 'dot_coeff': 1.8687594289211251, 'ehmi_coeff': 0.36066282711053776, 'dist_coeff': 0.3196793196332367},
    #'hiker': {'std': np.exp(-0.43926672), 'damping': np.exp(-2.33382131), 'scale': np.exp(-1.27401909), 'tau_threshold': np.exp(1.71610424), 'act_threshold': np.exp(0.35830021), 'pass_threshold': 1.67203308, 'dot_coeff': 5.60317922, 'ehmi_coeff': 0.4471767, 'dist_coeff': 0.40036073},
    #'hiker': {'std': 0.6280571068112765, 'damping': 0.8603387229027513, 'scale': 0.44441779584286256, 'tau_threshold': 2.732806932418672, 'act_threshold': 1.1120349553671918, 'pass_threshold': 1.376707085329192, 'dot_coeff': 1.7494757571462418, 'ehmi_coeff': 0.34796227462413976, 'dist_coeff': 0.4158610927992449},

    #'hiker': {'std': 0.6426495118602893, 'damping': 0.09796207737160328, 'scale': 0.27855577718398583, 'tau_threshold': 5.556581951608067, 'act_threshold': 1.4279091034209075, 'pass_threshold': 1.672125243901112, 'dot_coeff': 5.584069278738003, 'ehmi_coeff': 0.45141101837739356, 'dist_coeff': 0.4008133595156099}

    #'hiker': {'std': 0.6384974134032458, 'damping': 0.10186865847575911, 'scale': 0.2791830034375026, 'tau_threshold': 5.544952926633872, 'act_threshold': 1.4194944054263028, 'pass_threshold': 1.6722899150569444, 'dot_coeff': 5.5075638112727745, 'ehmi_coeff': 0.8482795240546034, 'dist_coeff': 0.4016305596474629}

    #'hiker': {'std': 0.6368401155044189, 'damping': 0.10504380918528354, 'scale': 0.2718230151392075, 'tau_threshold': 5.509627698960811, 'act_threshold': 1.4199126717680992, 'pass_threshold': 1.6722891886738138, 'dot_coeff': 5.5799566165845, 'ehmi_coeff': 0.47724126592004795, 'dist_coeff': 0.40163262102258657},

    # A first quick powell fit from the previous hiker values
    #'unified': {'std': 0.6558547131963791, 'damping': 0.09913879511497192, 'scale': 0.27224618351053537, 'tau_threshold': 5.517019096527725, 'act_threshold': 1.4436305690444342, 'pass_threshold': 1.6721177255112174, 'dot_coeff': 5.543668277443149, 'ehmi_coeff': 0.5304533310844964, 'dist_coeff': 0.39950975746351836},
    #'unified': {'std': 0.6747077314882771, 'damping': 1.4911827961638806, 'scale': 1.0605907628525664, 'tau_threshold': 2.561950744209917, 'act_threshold': 0.989999457038032, 'pass_threshold': 0.7177158069371236, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.22562395565228743}
    #'unified': {'std': 0.6714706667949863, 'damping': 1.3297068690871066, 'scale': 0.8980605549307248, 'tau_threshold': 2.6652088316253293, 'act_threshold': 0.9900001026088022, 'pass_threshold': 0.7180952350619795, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.21214932901594483},
    #'unified': {'std': 0.6712429538925937, 'damping': 1.3260607076620856, 'scale': 0.8992113108584929, 'tau_threshold': 2.6694337770093473, 'act_threshold': 0.9900000000187759, 'pass_threshold': 0.7180583191752358, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.21230195653562328}
    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.21231749517573556},

    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.2940552884232235, 'ehmi_coeff': 0, 'dist_coeff': 0.21231749517573556}
    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.2940554727554976, 'ehmi_coeff': 0, 'dist_coeff': 0.21231749517573556},

    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.2940554727554976, 'ehmi_coeff': 0.27190755157405394, 'dist_coeff': 0.21231749517573556},
    
    #'unified': {'std': 0.6862440192558461, 'damping': 1.223983663356895, 'scale': 0.6047489020410377, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000139199838, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.2206429111708295, 'ehmi_coeff': 0.34911133536812883, 'dist_coeff': 0.21414996598613342}
    
    # Constant speed only, with fixed keio
    #'unified': {'std': 0.726497450046289, 'damping': 0.5828631499887685, 'scale': 0.5723807755997636, 'tau_threshold': 2.8905817957021087, 'act_threshold': 1.5901754927214726, 'pass_threshold': 1.3446477949026612, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.342317232657446}
    #'unified': {'std': 0.7428544072653329, 'damping': 0.6370630838672563, 'scale': 0.6968811603124203, 'tau_threshold': 3.1131673216811158, 'act_threshold': 1.4700000000001503, 'pass_threshold': 1.2087810735174502, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.3461715480587001}
    #'unified': {'std': 0.7414992886209263, 'damping': 0.8288464497288173, 'scale': 0.8566811869814253, 'tau_threshold': 2.885489150938972, 'act_threshold': 1.3459994493368028, 'pass_threshold': 1.0111792830817665, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.3450656425781272},
    #'unified': {'std': 1.0, 'damping': 0.5, 'scale': 1.0, 'tau_threshold': 2.0, 'act_threshold': 1.0, 'pass_threshold': 1.0, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.0},
    # With new tau handling of the blocked trials
    #'unified': {'std': 0.7901846314196423, 'damping': 0.746895906952635, 'scale': 0.6764575296340535, 'tau_threshold': 2.234650305320731, 'act_threshold': 1.4700000005004261, 'pass_threshold': 1.1690402840993372, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.33280717909528784}
    #'unified': {'std': 0.7537887519909162, 'damping': 1.041642847452121, 'scale': 0.7998208638391029, 'tau_threshold': 2.233153875074307, 'act_threshold': 1.2299999999999638, 'pass_threshold': 0.9311804995030238, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.42475470128274023}
    
    # First iteration with only free dot_coeff
    #'unified': {'std': 0.7537887519909162, 'damping': 1.041642847452121, 'scale': 0.7998208638391029, 'tau_threshold': 2.233153875074307, 'act_threshold': 1.2299999999999638, 'pass_threshold': 0.9311804995030238, 'dot_coeff': 1.2682736344877221, 'ehmi_coeff': 0, 'dist_coeff': 0.42475470128274023},
    #'unified': {'std': 0.7537887519909162, 'damping': 1.041642847452121, 'scale': 0.7998208638391029, 'tau_threshold': 2.233153875074307, 'act_threshold': 1.2299999999999638, 'pass_threshold': 0.9311804995030238, 'dot_coeff': 1.2682361623624137, 'ehmi_coeff': 0.23947510490243312, 'dist_coeff': 0.42475470128274023}

    # Some mid of the run with full free parameterization
    #'unified': {'std': 0.7371838271814073, 'damping': 1.0385821829578816, 'scale': 0.6629842162971978, 'tau_threshold': 2.185455435064728, 'act_threshold': 1.2251442083024302, 'pass_threshold': 0.9661728261675379, 'dot_coeff': 1.2584974323810783, 'ehmi_coeff': 0.25238577013860924, 'dist_coeff': 0.4074043889521971}
    #'unified': {'std': 0.7424372666605178, 'damping': 0.9754312981640063, 'scale': 0.5909885393630063, 'tau_threshold': 2.1852519821467973, 'act_threshold': 1.2161517846923484, 'pass_threshold': 0.9655341477899285, 'dot_coeff': 1.2185182224709552, 'ehmi_coeff': 0.28129788574820386, 'dist_coeff': 0.4048724270579432}
    #'unified': {'std': 0.7210824769140926, 'damping': 0.8399760920484578, 'scale': 0.473255166371315, 'tau_threshold': 2.297556364408928, 'act_threshold': 1.1699999999997273, 'pass_threshold': 1.0015211556554193, 'dot_coeff': 1.2037864158316136, 'ehmi_coeff': 0.3256627194920388, 'dist_coeff': 0.37782960124387194}
    # Getting weird again...
    #'unified': {'std': 0.680663236276354, 'damping': 4.2463855335533245e-27, 'scale': 0.17682242248357646, 'tau_threshold': 3.4820810663186976, 'act_threshold': 1.1700000042206278, 'pass_threshold': 1.586319158425163, 'dot_coeff': 4.4736120544930085, 'ehmi_coeff': 0.6790887932271091, 'dist_coeff': 0.2904648152490044}
    #'unified': {'std': 0.8718651252295531, 'damping': 0.8228789721995772, 'scale': 0.6318861583019705, 'tau_threshold': 1.757234744466726, 'act_threshold': 1.5365982573580041, 'pass_threshold': 1.0460094109297184, 'dot_coeff': 1.427384168028474, 'ehmi_coeff': 0.47988147900383527, 'dist_coeff': 0.8871239618589292}

    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.2894392651525743, 'ehmi_coeff': 0.2676048639627061, 'dist_coeff': 0.21231749517573556}
    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.288879923787391, 'ehmi_coeff': 0.27558465363024653, 'dist_coeff': 0.21231749517573556},

    #'unified': {'std': 0.6603592740326082, 'damping': 1.1052455453328458, 'scale': 0.5141555441698754, 'tau_threshold': 2.681525975300402, 'act_threshold': 0.9900000000012792, 'pass_threshold': 0.7843853329736621, 'dot_coeff': 1.2279377218791985, 'ehmi_coeff': 0.373094808195673, 'dist_coeff': 0.20647930981743343}
    #'unified': {'std': 0.69746300229369, 'damping': 0.46613114379319825, 'scale': 0.3853993922254003, 'tau_threshold': 3.6592793898164855, 'act_threshold': 1.3499999966920435, 'pass_threshold': 1.31085488, 'dot_coeff': 2.68806523, 'ehmi_coeff': 0.42923702, 'dist_coeff': 0.27769681}
    
    # KEIO-only, constant speed only
    #'unified':  {'std': 0.6977129806038627, 'damping': 1.3782352987426398, 'scale': 0.8429963353952452, 'tau_threshold': 1.9035952542119765, 'act_threshold': 0.984344221363203, 'pass_threshold': -0.009199830983710197, 'dot_coeff': 1e-9, 'ehmi_coeff': 0, 'dist_coeff': 0.6488016222745131}
    # dot_coeff fitted independently
    #'unified': {'std': 0.6977129806038627, 'damping': 1.3782352987426398, 'scale': 0.8429963353952452, 'tau_threshold': 1.9035952542119765, 'act_threshold': 0.984344221363203, 'pass_threshold': -0.009199830983710197, 'dot_coeff': 0.5400225892173597, 'ehmi_coeff': 0, 'dist_coeff': 0.6488016222745131}
    
    # Full free param fit to KEIO
    #'unified': {'std': 0.6884581019986444, 'damping': 1.5507755865637733, 'scale': 0.6014947680979241, 'tau_threshold': 1.786246422626752, 'act_threshold': 0.9164873080720805, 'pass_threshold': -0.07068913423131726, 'dot_coeff': 0.6434015648391005, 'ehmi_coeff': 0, 'dist_coeff': 0.777781366472576}
    #'unified': {'std': 0.6880891949348873, 'damping': 1.5665591519741577, 'scale': 0.6037404637215124, 'tau_threshold': 1.7835700314266882, 'act_threshold': 0.9151360985627606, 'pass_threshold': -0.07068913421388402, 'dot_coeff': 0.6411745992873026, 'ehmi_coeff': 0.5, 'dist_coeff': 0.7762157904579379}
    #-399.50826504948475
    #'unified': {'std': 0.6038421478875342, 'damping': 2.247573697572023, 'scale': 0.6441694353318161, 'tau_threshold': 1.5081177218235644, 'act_threshold': 0.7842575645593984, 'pass_threshold': -0.1099999996790976, 'dot_coeff': 0.5752662551979117, 'ehmi_coeff': 0.5, 'dist_coeff': 0.77791232525027}
    #'unified': {'std': 0.5992295175443879, 'damping': 2.206024649270368, 'scale': 0.6034675726826785, 'tau_threshold': 1.4664311180073841, 'act_threshold': 0.7844550749937309, 'pass_threshold': -0.1099999996322923, 'dot_coeff': 0.5728792057070659, 'ehmi_coeff': 0.5, 'dist_coeff': 0.7775901533478482} #-399.3551751046223
    
    
    # Fit only pass threshold to hiker
    #'unified': {'std': 0.6880891949348873, 'damping': 1.5665591519741577, 'scale': 0.6037404637215124, 'tau_threshold': 1.7835700314266882, 'act_threshold': 0.9151360985627606, 'pass_threshold': 0.6526031285284474, 'dot_coeff': 0.6411745992873026, 'ehmi_coeff': 0.5, 'dist_coeff': 0.7762157904579379}
    
    # Some mid-of-optimization stuff
    #'unified': {'std': 0.6378917671331962, 'damping': 1.3094120471486868, 'scale': 0.5366755373566473, 'tau_threshold': 2.5449256246780716, 'act_threshold': 0.9328895826640872, 'pass_threshold': 0.38511415788509556, 'dot_coeff': 0.9675905511781857, 'ehmi_coeff': 0.35698493728498476, 'dist_coeff': 0.04917540839195181}
    #'unified': {'std': 0.6569462641896981, 'damping': 0.8991304858288773, 'scale': 0.448500772120246, 'tau_threshold': 2.836317012102674, 'act_threshold': 1.0499930079742774, 'pass_threshold': 0.1606736448009839, 'dot_coeff': 1.5213214576379954, 'ehmi_coeff': 0.41887059935472765, 'dist_coeff': 0.3172478077457032}
    #'unified': {'std': 0.7078287012725756, 'damping': 0.7990847222026269, 'scale': 0.5870407478850296, 'tau_threshold': 3.1652111466428825, 'act_threshold': 1.173541779637534, 'pass_threshold': 0.2342047425801052, 'dot_coeff': 2.0669310844752853, 'ehmi_coeff': 0.3754702100469714, 'dist_coeff': 0.38305142824262445}
    # Full free without ehmi, mid-optimization, -8177.111908268538 
    #'unified': {'std': 0.5861015695713019, 'damping': 2.2881771610251658, 'scale': 0.5474824518937139, 'tau_threshold': 2.051828725048746, 'act_threshold': 0.7332946416938594, 'pass_threshold': 0.2831412296181032, 'dot_coeff': 1.2977399850131366, 'ehmi_coeff': 0.5, 'dist_coeff': 0.7035002474403371}
    #'unified': {'std': 0.6275717494399937, 'damping': 1.3002793035212736, 'scale': 0.5301530656966875, 'tau_threshold': 2.6771001396634584, 'act_threshold': 0.8843450284334016, 'pass_threshold': 0.3511018282085151, 'dot_coeff': 1.1037716833506162, 'ehmi_coeff': 0.0, 'dist_coeff': 0.19462017060079942} #-7816.386931963873
    # Going insane again
    #'unified': {'std': 0.6812593161594435, 'damping': 0.5201168534657028, 'scale': 0.40718793863965597, 'tau_threshold': 3.578314005111645, 'act_threshold': 1.2434773355925064, 'pass_threshold': 0.8655316312010426, 'dot_coeff': 2.5891293432038394, 'ehmi_coeff': 0.0, 'dist_coeff': 0.34024495873256055} #-7476.630423498658

    # HIKER pass_threshold and ehmi_coeff from KEIO fit
    #'unified': {'std': 0.5992295175443879, 'damping': 2.206024649270368, 'scale': 0.6034675726826785, 'tau_threshold': 1.4664311180073841, 'act_threshold': 0.7844550749937309, 'pass_threshold': 0.2791404411912775, 'dot_coeff': 0.5728792057070659, 'ehmi_coeff': 0.4551496580971471, 'dist_coeff': 0.7775901533478482} #10752.451062834896
    

    # HIKER and KEIO simultaneous fit for constant speeds only. 100 basinhopping iterations, loglik -2715.87108922
    #'unified': {'std': 0.7154420728719368, 'damping': 0.7135802228585579, 'scale': 0.6508319192092222, 'tau_threshold': 2.9990588879537867, 'act_threshold': 1.3503757003891326, 'pass_threshold': 0.742214654641068, 'dot_coeff': 0.5728792057070659, 'ehmi_coeff': 0.5, 'dist_coeff': 0.3313723761689591},
    # Dot coeff fitted to previous
    #'unified': {'std': 0.7154420728719368, 'damping': 0.7135802228585579, 'scale': 0.6508319192092222, 'tau_threshold': 2.9990588879537867, 'act_threshold': 1.3503757003891326, 'pass_threshold': 0.742214654641068, 'dot_coeff': 2.0610960154899263, 'ehmi_coeff': 1e-6, 'dist_coeff': 0.3313723761689591}
    # ehmi coeff fitted to previous
    #'unified': {'std': 0.7154420728719368, 'damping': 0.7135802228585579, 'scale': 0.6508319192092222, 'tau_threshold': 2.9990588879537867, 'act_threshold': 1.3503757003891326, 'pass_threshold': 0.742214654641068, 'dot_coeff': 2.0610960154899263, 'ehmi_coeff': 0.2872315786358714, 'dist_coeff': 0.3313723761689591}

    # HIKER only, constant speed only
    #'unified': {'std': 0.6243259056065988, 'damping': 1.2282701409740862, 'scale': 0.6020444557245085, 'tau_threshold': 2.426798484805438, 'act_threshold': 0.9300000000000214, 'pass_threshold': 0.416198091716651, 'dot_coeff': 1e-09, 'ehmi_coeff': 0, 'dist_coeff': 0.48404506973434824} #2721.3616694930306
    # Prev with dot_coeff optimized
    #'unified': {'std': 0.6243259056065988, 'damping': 1.2282701409740862, 'scale': 0.6020444557245085, 'tau_threshold': 2.426798484805438, 'act_threshold': 0.9300000000000214, 'pass_threshold': 0.416198091716651, 'dot_coeff': 1.3319437693617528, 'ehmi_coeff': 0, 'dist_coeff': 0.48404506973434824}
    # Prev with ehmi_coeff optimized
    #'unified': {'std': 0.6243259056065988, 'damping': 1.2282701409740862, 'scale': 0.6020444557245085, 'tau_threshold': 2.426798484805438, 'act_threshold': 0.9300000000000214, 'pass_threshold': 0.416198091716651, 'dot_coeff': 1.3319437693617528, 'ehmi_coeff': 0.3011107895169946, 'dist_coeff': 0.48404506973434824}
    # Prev with full free parameterization
    #'unified': {'std': 0.656377694293709, 'damping': 1.122737257849244, 'scale': 0.46737924590077873, 'tau_threshold': 2.4978711105790845, 'act_threshold': 0.943499017711252, 'pass_threshold': 0.4373563301212072, 'dot_coeff': 0.9790717773232571, 'ehmi_coeff': 0.3569624118902206, 'dist_coeff': 0.21717397761054827}

    # Full Keio, first basinghopping iteration
    #'keio_uk': {'std': 0.5343251552162854, 'damping': 2.582538225032225, 'scale': 0.561655673298275, 'tau_threshold': 1.183889032468715, 'act_threshold': 0.6971501469537572, 'pass_threshold': -0.17627817652058925, 'dot_coeff': 0.4878925491203472, 'ehmi_coeff': 0.0, 'dist_coeff': 0.7701321073773191}, #399.6714197520489
    
    # Full Keio with new time constant formulation and no unnecessary logbarriers.
    # WEIRD!!!
    #'keio_uk': {'std': 0.6596074501365182, 'damping': 8.050502783469992e+205, 'scale': 0.3074943776061263, 'tau_threshold': 2.196541120462038, 'act_threshold': 1.0, 'pass_threshold': 0.01802908773814163, 'dot_coeff': -0.945639683315949, 'ehmi_coeff': 0.0, 'dist_coeff': 0.4546431509219718}, # -397.04508957804666
    
    # Full Keio basinhopping with no unnecessary logbarriers and fixed act_threshold
    #'keio_uk': {'std': 0.7683371967321696, 'damping': 1.2361164021518585, 'scale': 0.6022794383801227, 'tau_threshold': 2.0764018144820233, 'act_threshold': 1.0, 'pass_threshold': -0.06978790574678727, 'dot_coeff': 0.716516036380512, 'ehmi_coeff': 0.0, 'dist_coeff': 0.7552155764222939}, # -401.51522150074277
    # Full Keio basinhopping with 14 iters with no unnecessary logbarriers
    #'keio_uk': {'std': 0.6743200476268959, 'damping': 1.7375574919520165, 'scale': 0.582044434824494, 'tau_threshold': 1.7784420654168027, 'act_threshold': 0.8623734414082952, 'pass_threshold': -0.14332747696286155, 'dot_coeff': 0.6171365192574244, 'ehmi_coeff': 0.0, 'dist_coeff': 0.7537096443402346}, # -400.9477469586112
    # More iterations to above
    #'keio_uk': {'std': 0.6180859388145251, 'damping': 1.926971433004895, 'scale': 0.6020413272036864, 'tau_threshold': 1.5244315485778317, 'act_threshold': 0.8188728265419005, 'pass_threshold': -0.13831946551367255, 'dot_coeff': 0.5931163941352746, 'ehmi_coeff': 0.0, 'dist_coeff': 0.7849626818288308}, # -400.9398436713008
    # More iterations to above
    #'keio_uk': {'std': 0.6195190762900944, 'damping': 1.929186500212442, 'scale': 0.6010065333354125, 'tau_threshold': 1.5240523644347281, 'act_threshold': 0.8187184667744724, 'pass_threshold': -0.13420139794611713, 'dot_coeff': 0.5899926559079496, 'ehmi_coeff': 0.0, 'dist_coeff': 0.7812059668687735}, #-400.932563397718
    # More iterations
    'keio_uk': {'std': 0.6194137333928834, 'damping': 1.9306102005519206, 'scale': 0.6000089156362227, 'tau_threshold': 1.5249170232823437, 'act_threshold': 0.8183601344001614, 'pass_threshold': -0.14263846715640696, 'dot_coeff': 0.5873341694195724, 'ehmi_coeff': 0.0, 'dist_coeff': 0.778726921288107}, #-400.9303668742628

    # pass_threshold and ehmi_coeff fitted to HIKER
    #'hiker': {'std': 0.5343251552162854, 'damping': 2.582538225032225, 'scale': 0.561655673298275, 'tau_threshold': 1.183889032468715, 'act_threshold': 0.6971501469537572, 'pass_threshold': 0.28361016189473476, 'dot_coeff': 0.4878925491203472, 'ehmi_coeff': 0.45785997539732376, 'dist_coeff': 0.7701321073773191}, # -10850.709835672875
    # With proper compensation for the rear-front-coordinate mess
    #'hiker': {'std': 0.5343251552162854, 'damping': 2.582538225032225, 'scale': 0.561655673298275, 'tau_threshold': 1.183889032468715, 'act_threshold': 0.6971501469537572, 'pass_threshold': 0.3097080802703172, 'dot_coeff': 0.4878925491203472, 'ehmi_coeff': 0.4211867042001631, 'dist_coeff': 0.7701321073773191} #-10755.027885100471
    # First basinhopping iteration, full dataset with fixed eHMI
    #'hiker': {'std': 0.5343251552162854, 'damping': 2.582538225032225, 'scale': 0.561655673298275, 'tau_threshold': 1.183889032468715, 'act_threshold': 0.6971501469537572, 'pass_threshold': 0.3119566717182752, 'dot_coeff': 0.4878925491203472, 'ehmi_coeff': 0.569211027600194, 'dist_coeff': 0.7701321073773191} #-10698.58129193798
    # Few basinhoppings for FH only
    #'hiker': {'std': 0.5343251552162854, 'damping': 2.582538225032225, 'scale': 0.561655673298275, 'tau_threshold': 1.183889032468715, 'act_threshold': 0.6971501469537572, 'pass_threshold': 0.3112119482441364, 'dot_coeff': 0.4878925491203472, 'ehmi_coeff': 0.9526989219987926, 'dist_coeff': 0.7701321073773191} #-7301.504166653411
    # Full basinhopping, FH only
    'hiker': {'std': 0.5343251552162854, 'damping': 2.582538225032225, 'scale': 0.561655673298275, 'tau_threshold': 1.183889032468715, 'act_threshold': 0.6971501469537572, 'pass_threshold': 0.30895195097474315, 'dot_coeff': 0.4878925491203472, 'ehmi_coeff': 0.9526989030248043, 'dist_coeff': 0.7701321073773191} #-7301.504166653412
    }
# Keio with eHMI estimated from HIKER
vddm_params['unified'] = {**vddm_params['keio_uk'], 'ehmi_coeff': vddm_params['hiker']['ehmi_coeff']}

tdm_params = {
    #'keio_uk': {'thm': 1.4822542031746122, 'ths': 0.4160580398784046, 'lagm': -0.018037627273216346, 'lags': 0.5394395959794362, 'pass_th': -0.07255736056398886},
    'keio_japan': {'thm': 1.7737959051473795, 'ths': 0.4270654760140321, 'lagm': 0.15978825057281343, 'lags': 0.5678380232026445, 'pass_th': -0.015453144629448029},
    #'hiker': {'thm': 1.58413762002706, 'ths': 0.30787584232846543, 'lagm': 0.18325303499450624, 'lags': 0.24160198818000903, 'pass_th': 0.9960908329751528},
    #'hiker': {'thm': 1.5152455089318642, 'ths': 0.32602324226593116, 'lagm': 0.1245117672647987, 'lags': 0.6021881227736422, 'pass_th': 0.7477010070264796, 'dot_coeff': 2.6937678922001163, 'ehmi_coeff': 0.2805016925833088}
    #'hiker': {'thm': 1.5442566242657068, 'ths': 0.3166462662796281, 'lagm': 0.20509904820328212, 'lags': 0.5482381938105031, 'pass_th': 0.8428738656551962, 'dot_coeff': 3.207119275055747, 'ehmi_coeff': 0.36038229654487886, 'dist_coeff': 0.2233817144842581}

    'hiker': {'thm': 1.5589096041551214, 'ths': 0.31723512786804214, 'lagm': 0.22346854221177187, 'lags': 0.5274406179540562, 'pass_th': 0.8792264031648431, 'dot_coeff': 3.2118358447257203, 'ehmi_coeff': 0.91032775986075, 'dist_coeff': 0.19840058606111935},
    
    # A first quick powell fit from the previous hiker values
    #'unified': {'thm': 1.5587824305997904, 'ths': 0.3271553709034499, 'lagm': 0.22576407802136994, 'lags': 0.5386053750104438, 'pass_th': 0.8792264031911448, 'dot_coeff': 3.21409347065363, 'ehmi_coeff': 0.8082487302244402, 'dist_coeff': 0.19834381246597568}

    # Full basinhopping to HIKER constant speed only
    #'unified': {'thm': 1.4343609857798998, 'ths': 0.3525943320057186, 'lagm': 0.36012224052896186, 'lags': 0.20523575122335908, 'pass_th': 0.8407536177805159, 'dot_coeff': 1.0, 'ehmi_coeff': 1e-6, 'dist_coeff': 0.3401497300077063}
    # dot_coeff to previous
    #'unified': {'thm': 1.4343609857798998, 'ths': 0.3525943320057186, 'lagm': 0.36012224052896186, 'lags': 0.20523575122335908, 'pass_th': 0.8407536177805159, 'dot_coeff': 4.842686097140675, 'ehmi_coeff': 1e-06, 'dist_coeff': 0.3401497300077063}
    
    # Initial for Keio fit
    #'unified': {'thm': np.log(3.0), 'ths': 1/np.sqrt(6), 'lagm': np.log(1.0), 'lags': 1/np.sqrt(6), 'pass_th': 0.0, 'dot_coeff': 1e-6, 'ehmi_coeff': 1e-06, 'dist_coeff': 1e-6}
    # Full Keio basinhopping fit
    'keio_uk': {'thm': 1.4301503191776679, 'ths': 0.4047969673928879, 'lagm': 0.016608210303132295, 'lags': 0.6036584682907242, 'pass_th': -0.07599470114085044, 'dot_coeff': 1.40164385976053, 'ehmi_coeff': 1e-06, 'dist_coeff': 0.2518762000227112},

    # pass_th and ehmi_coeff fitted to HIKER
    #'hiker': {'thm': 1.4301503191776679, 'ths': 0.4047969673928879, 'lagm': 0.016608210303132295, 'lags': 0.6036584682907242, 'pass_th': 0.28063729862263254, 'dot_coeff': 1.40164385976053, 'ehmi_coeff': 1.3789088107061784, 'dist_coeff': 0.2518762000227112}, # -11555.08182928335
    #'hiker': {'thm': 1.4301503191776679, 'ths': 0.4047969673928879, 'lagm': 0.016608210303132295, 'lags': 0.6036584682907242, 'pass_th': 0.2811319315973182, 'dot_coeff': 1.40164385976053, 'ehmi_coeff': 1.0109547383295463, 'dist_coeff': 0.2518762000227112} # -11350.733197340674
    # Full fixed dataset -11303.8973577955
    #'hiker': {'thm': 1.4301503191776679, 'ths': 0.4047969673928879, 'lagm': 0.016608210303132295, 'lags': 0.6036584682907242, 'pass_th': 0.28116492718247144, 'dot_coeff': 1.40164385976053, 'ehmi_coeff': 1.1959845922577328, 'dist_coeff': 0.2518762000227112}
    # For FH only -7649.858412900177
    'hiker': {'thm': 1.4301503191776679, 'ths': 0.4047969673928879, 'lagm': 0.016608210303132295, 'lags': 0.6036584682907242, 'pass_th': 0.37342587427273055, 'dot_coeff': 1.40164385976053, 'ehmi_coeff': 2.045868748819189, 'dist_coeff': 0.2518762000227112}
}
# Keio with eHMI fitted to HIKER
tdm_params['unified'] = {**tdm_params['keio_uk'], 'ehmi_coeff': tdm_params['hiker']['ehmi_coeff']}


def mangle_tau(traj, traj_b=None, pass_threshold=0.0, dist_coeff=0.0, dot_coeff=0.0, ehmi_coeff=0.0, **kwargs):
    distance = traj['distance'].copy()
    tau = distance/traj['speed']
    tau_dot = np.gradient(tau, DT)
    tau_dot[~np.isfinite(tau_dot)] = 0 # Fix for when speed goes to zero

    if traj_b is not None:
        not_passed = traj_b["distance"] > 0
        distance[not_passed] = (distance - traj_b["distance"])[not_passed]
    
        tau = distance/traj['speed']
        #tau_dot = np.gradient(tau, DT)
    #tau_dot = traj['tau_dot']
    prior_tau = distance/(50/3.6)
    
    passed = tau < pass_threshold
    tau = dist_coeff*(prior_tau - tau) + tau + dot_coeff*(tau_dot + 1) + ehmi_coeff*traj['ehmi']
    tau[traj['speed'] == 0] = np.inf
    tau[passed] = np.inf
    return tau

def model_params(params):
    return {k: v for k, v in params.items() if k not in ('pass_threshold', 'dist_coeff', 'dot_coeff', 'ehmi_coeff')}

actgrid = Grid1d(-3.0, 3.0, 100)
def fit_vdd(trials, dt):
    init = dict(
           std=1.0,
           damping=1.0,
           scale=1.0,
           tau_threshold=2.5,
           act_threshold=1.0,
           pass_threshold=0.0
           )


    spec = dict(
        std=            (init['std'], logbarrier),
        damping=        (init['damping'], logbarrier),
        scale=          (init['scale'], logbarrier,),
        tau_threshold=  (init['tau_threshold'], logbarrier),
        act_threshold=  (init['act_threshold'], logbarrier,),
        pass_threshold= (0.0,),
        dot_coeff=      (0.001, logbarrier)
            )
    
    def loss(**params):
        loss = 0.0
        model = Vddm(dt=dt, **model_params(params))
        for traj, cts in trials:
            tau = mangle_tau(traj, **params)
            mylik = model.decisions(actgrid, tau).loglikelihood(cts)
            loss -= mylik
        return loss
    
    def cb(x, f, accept):
        print(f, accept, x)
    return minimizer(loss,
            method='powell', #options={'maxiter': 1}
            )(**spec)
    
def fit_blocker_vdd(trials, dt, init=vddm_params['unified']):
    spec = dict(
        std=            (init['std'], logbarrier, fixed),
        damping=        (init['damping'], logbarrier, fixed),
        scale=          (init['scale'], logbarrier, fixed),
        tau_threshold=  (init['tau_threshold'], logbarrier, fixed),
        act_threshold=  (init['act_threshold'], logbarrier, fixed),# (init['act_threshold'], logbarrier),
        pass_threshold= (init['pass_threshold']),
        dot_coeff=      (init['dot_coeff'],fixed),
        ehmi_coeff=      (init['ehmi_coeff']),
        dist_coeff=     (init['dist_coeff'],fixed)
            )
    
    def loss(**params):
        loss = 0.0
        model = Vddm(dt=dt, **model_params(params))
        #print(model.tau_threshold)
        for traj, traj_b, cts in trials:
            tau = mangle_tau(traj, traj_b, **params)
            tau_b = mangle_tau(traj_b, **params)
            pdf = model.blocker_decisions(actgrid, tau, tau_b)
            myloss = pdf.loglikelihood(cts - traj.time[0], slack=np.finfo(float).eps)
            loss -= myloss
        print(loss)
        return loss
    def cb(x, f, accept):
        print(f, accept, x)
    
    return minimizer(loss, method='powell')(**spec)
    #return minimizer(loss, scipy.optimize.basinhopping, T=10.0,
    #        callback=cb, minimizer_kwargs={'method': 'powell'}
    #        #method='powell', #options={'maxiter': 1}
    #        )(**spec)

def fit_tdm(trials, dt):
    spec = dict(
        thm=            (np.log(3.0),),
        ths=            (np.sqrt(1/6), logbarrier),
        lagm=            (np.log(0.3),),
        lags=            (np.sqrt(1/6), logbarrier),
        pass_th=          (1.0),
        dot_coeff=      (0.001, logbarrier)
            )
    
    def loss(**params):
        lik = 0
        model = Tdm(**model_params(params))
        for traj, rts in trials:
            tau = mangle_tau(traj, **params)
            pdf = model.decisions(tau, dt)
            lik += pdf.loglikelihood(rts, np.finfo(float).eps)
        return -lik
    return minimizer(loss, method='powell', #options={'maxiter': 1}
            )(**spec)

def fit_blocker_tdm(trials, dt, init=tdm_params['unified']):
    spec = dict(
        thm=            (init['thm'],),
        ths=            (init['ths'], logbarrier),
        lagm=            (init['lagm'],),
        lags=            (init['lags'], logbarrier),
        pass_th=          (init['pass_th']),
        dot_coeff=      (init['dot_coeff'],),
        ehmi_coeff=      (init['ehmi_coeff'],),
        dist_coeff=     (init['dist_coeff'],),
            )
    
    def loss(**params):
        lik = 0
        model = Tdm(**model_params(params))
        for traj, traj_b, rts in trials:
            tau = mangle_tau(traj, traj_b, **params)
            tau_b = mangle_tau(traj_b, **params)
            pdf = model.blocker_decisions(tau, tau_b, dt)
            lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
        return -lik
    
    def cb(x, f, accept):
        print(f, accept, x)
    
    #return minimizer(loss, method='powell')(**spec)
    return minimizer(loss, scipy.optimize.basinhopping, T=10.0,
            callback=cb, minimizer_kwargs={'method': 'powell'}
            #method='powell', #options={'maxiter': 1}
            )(**spec)

def fit_unified_vddm(trials, dt, init=vddm_params['keio_uk']):
    spec = dict(
        std=            (init['std'], logbarrier),
        damping=        (init['damping'],),
        scale=          (init['scale'], logbarrier),
        tau_threshold=  (init['tau_threshold']),
        act_threshold=  (init['act_threshold'], logbarrier),
        pass_threshold= (init['pass_threshold'],),
        dot_coeff=      (init['dot_coeff'],),
        ehmi_coeff=      (init['ehmi_coeff'],fixed),
        dist_coeff=     (init['dist_coeff'],)
    )
    
    """
    spec = dict(
        std=            (1.0, logbarrier),
        damping=        (0.0,),
        scale=          (1.0, logbarrier),
        tau_threshold=  (2.0,),
        act_threshold=  (1.0, logbarrier),
        pass_threshold= (0.0,),
        dot_coeff=      (0.0,),
        ehmi_coeff=      (0.0,fixed),
        dist_coeff=     (0.0,)
    )
    """
    
    bestlik = -np.inf
    def loss(**params):
        lik = 0
        
        model = Vddm(dt=dt, **model_params(params))
        for trial in trials:
            if len(trial) == 3:
                traj, traj_b, rts = trial
                tau = mangle_tau(traj, traj_b, **params)
                tau_b = mangle_tau(traj_b, **params)
                pdf = model.blocker_decisions(actgrid, tau, tau_b)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
            else:
                traj, rts = trial
                tau = mangle_tau(traj, **params)
                pdf = model.decisions(actgrid, tau)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
        nonlocal bestlik
        if lik != lik:
            print("NANANAN")
            print(params)
        if lik > bestlik:
            bestlik = lik
            print(params)
            print(lik)
        return -lik
    
    iters = 0
    def cb(x, f, accept):
        nonlocal iters
        iters += 1
        print("Basinhopping iter done", iters)
        return
        print(kwopt.unmangle(spec, x))
        print(f, accept)
    
    """
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.utils import use_named_args
    from skopt.space import Real
    dims = [
        Real(name="std", low=0, high=2.0),
        Real(name="damping", low=1e-30, high=(1 - 1e-30)),
        Real(name="scale", low=0, high=2.0),
        Real(name="tau_threshold", low=0.0, high=3.0),
        Real(name="act_threshold", low=0.0, high=2.0),
        Real(name="pass_threshold", low=0.0, high=2.0),
        Real(name="dot_coeff", low=0.0, high=2.0),
        Real(name="ehmi_coeff", low=0.0, high=1.0),
        Real(name="dist_coeff", low=0.0, high=1.0),
            ]
    x0 = [init[d.name] for d in dims]
    wtf = forest_minimize(use_named_args(dimensions=dims)(loss), dims, verbose=True, n_calls=1000,
            #x0=x0,
            acq_func='PI',
            )
    print(wtf)
    return
    """
    return minimizer(loss, scipy.optimize.basinhopping, T=1.0,
            callback=cb, minimizer_kwargs={'method': 'powell'}
            #method='powell', #options={'maxiter': 1}
            )(**spec)
    
    #return minimizer(loss, method='powell')(**spec)
    return minimizer(loss, scipy.optimize.basinhopping, T=10.0,
            callback=cb, minimizer_kwargs={'method': 'powell'}
            #method='powell', #options={'maxiter': 1}
            )(**spec)
    

def fit_unified_tdm(trials, dt, init=tdm_params['unified']):
    spec = dict(
        thm=            (init['thm'],fixed),
        ths=            (init['ths'], logbarrier,fixed),
        lagm=            (init['lagm'],fixed),
        lags=            (init['lags'], logbarrier,fixed),
        pass_th=          (init['pass_th'],),
        dot_coeff=      (init['dot_coeff'],logbarrier,fixed),
        ehmi_coeff=      (init['ehmi_coeff'],logbarrier,),
        dist_coeff=     (init['dist_coeff'],logbarrier,fixed),
            )
    
    bestlik = -np.inf
    def loss(**params):
        nonlocal bestlik
        lik = 0
        model = Tdm(**model_params(params))
        for trial in trials:
            if len(trial) == 3:
                traj, traj_b, rts = trial
                tau = mangle_tau(traj, traj_b, **params)
                tau_b = mangle_tau(traj_b, **params)
                pdf = model.blocker_decisions(tau, tau_b, dt)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
            else:
                traj, rts = trial
                tau = mangle_tau(traj, **params)
                pdf = model.decisions(tau, dt)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
            
        if lik != lik:
            print(lik, params)
        if lik > bestlik:
            print(lik)
            print(params)
            bestlik = lik
        return -lik
    
    #def cb(*args):
    #    print(args)

    return minimizer(loss, scipy.optimize.basinhopping, T=10.0,
            callback=cb, minimizer_kwargs={'method': 'powell'}
            #method='powell', #options={'maxiter': 1}
            )(**spec)
    #return minimizer(loss, method='powell')(**spec)

def get_hiker_trials(include_constants=True, include_decels=True, include_ehmi=True, include_ehmi_controls=True, include_splb=False, include_fh=True):
    data = pd.read_csv('hiker_cts.csv')
    data = data.query('braking_condition <= 3')
    if not include_constants:
        data = data.query('is_braking == True')
    if not include_decels:
        data = data.query('is_braking == False')
    if not include_ehmi:
        data = data.query('has_ehmi == False')
    if not include_ehmi_controls:
        data = data.query('ehmi_type != "none"')
    if not include_splb:
        data = data.query("ehmi_type != 'SPLB'")
    if not include_fh:
        data = data.query("ehmi_type != 'FH'")
    trials = []
    for g, d in data.groupby(['time_gap', 'speed', 'is_braking', 'has_ehmi']):
        time_gap, speed, is_braking, has_hmi = g
        
        crossing_times = d.crossing_time.values
        # The original values were computed with lead vehicle rear coordinates.
        # Compensate to the zero time when the vehicle front crosses the zero.
        # This happens vehicle_length/speed seconds earlier than for the rear,
        # so in reference the crossing times are later.
        crossing_times += vehicle_length/speed 
        crossing_times[~np.isfinite(crossing_times)] = np.inf

        traj, lead_traj = get_trajectory(time_gap, speed, is_braking, has_hmi)
        
        #traj = np.rec.fromarrays((-lag_x,tau_lag, np.gradient(tau_lag, DT), ehmi), names="distance,tau,tau_dot,ehmi")
        trials.append((traj, lead_traj, crossing_times))

    return trials

def get_keio_trials(country='uk', include_constants=True, include_decels=True, **kwargs):
    all_trajectories = pd.read_csv('d2p_trajectories.csv').rename(columns={'time_c': 'time'})
    all_responses = pd.read_csv(f'd2p_cross_times_{country}.csv')
    all_responses[["subject_id", "trial_number"]] = all_responses.unique_ID.str.split("_", 1, expand=True)
    responses = dict(list(all_responses.groupby('trial_id')))

    trials = {}
    trial_taus = {}
    subj_trial_responses = {}
    
    all_trajectories = all_trajectories.query("trial_n > 2 and trial_n <= 16")

    for trial, traj in all_trajectories.groupby('trial_n'):
        has_decel = np.std(traj['speed']) > 0.1
        if has_decel and not include_decels:
            continue
        if not has_decel and not include_constants:
            continue
        resp = responses[trial]['cross_time'].values
        #tau = traj['tau'].values
        traj['ehmi'] = False
        # Recompute these as they are mangled in the original
        traj['tau'] = traj['distance'].values/traj['speed'].values
        traj['tau_dot'] = np.gradient(traj.tau.values, DT)
        trials[trial] = (traj.to_records(), resp)
    
    return list(trials.values())

def fit_hiker_and_keio():
    subset = dict(
        include_ehmi        = False,
        include_decels      = True,
        include_constants   = True,
        include_splb        = False,
        include_fh          = False
        )
    trials = []
    #trials += get_hiker_trials(**subset)
    trials += get_keio_trials(**subset)

    fit = fit_unified_vddm(trials, DT, init=vddm_params['keio_uk'])
    #fit = fit_unified_tdm(trials, DT, init=tdm_params['keio_uk'])
    #print(fit)

    
def fit_hiker():
    data = pd.read_csv('hiker_cts.csv')

    #data = data.query('braking_condition == 1')
    data = data.query('braking_condition <= 3')
    
    leader_start = 100
    DT = 1/30
    trials = []
    for g, d in data.groupby(['time_gap', 'speed', 'is_braking', 'has_ehmi']):
        time_gap, speed, is_braking, has_hmi = g
        
        """
        starttime = -leader_start/speed
        endtime = starttime + 20
        if not is_braking:
            endtime = time_gap

        ts = np.arange(starttime, endtime, DT)
        lag_x, lag_speed, (t_brake, t_stop) = hikersim.simulate_trajectory(ts, time_gap, speed, is_braking)

        
        tau_lag = -lag_x/lag_speed
        tau_lag[~np.isfinite(tau_lag)] = 1e6
        lead_dist = leader_start - (ts - starttime)*speed
        tau_lead = lead_dist/speed
        tau_lead[~np.isfinite(tau_lead)] = 1e6
        
        crossing_times = d.crossing_time.values - starttime
        crossing_times[~np.isfinite(crossing_times)] = np.inf
        
        lead_traj = np.rec.fromarrays(
                (lead_dist,tau_lead, np.gradient(tau_lead, DT), np.zeros(len(ts))),
                names="distance,tau,tau_dot,ehmi")
        
        ehmi = np.zeros(len(ts))
        if has_hmi:
            ehmi[ts >= t_brake] = 1.0
        """
        
        crossing_times = d.crossing_time.values
        crossing_times[~np.isfinite(crossing_times)] = np.inf

        traj, lead_traj = get_trajectory(time_gap, speed, is_braking, has_hmi)
        
        #traj = np.rec.fromarrays((-lag_x,tau_lag, np.gradient(tau_lag, DT), ehmi), names="distance,tau,tau_dot,ehmi")
        trials.append((traj, lead_traj, crossing_times))

    vdd_fit = fit_blocker_vdd(trials, DT)
    vdd_params = vdd_fit.kwargs
    print("VDD")
    print(vdd_fit)
    
    #tdm_fit = fit_blocker_tdm(trials, DT)
    #tdm_params = tdm_fit.kwargs
    #print("TDM")
    #print(tdm_fit)


def fit_keio(country='uk'):
    all_trajectories = pd.read_csv('d2p_trajectories.csv')
    all_responses = pd.read_csv(f'd2p_cross_times_{country}.csv')
    all_responses[["subject_id", "trial_number"]] = all_responses.unique_ID.str.split("_", 1, expand=True)
    responses = dict(list(all_responses.groupby('trial_id')))

    trials = {}
    trial_taus = {}
    subj_trial_responses = {}

    for trial, traj in all_trajectories.groupby('trial_n'):
        
        if np.std(traj.speed) > 0.01:
            # Hmm.. why only constant speed?
            continue
        resp = responses[trial]['cross_time'].values
        #tau = traj['tau'].values
        traj['ehmi'] = False
        tau = traj['distance'].values/traj['speed'].values
        trials[trial] = (traj.to_records(), resp)

    for s, sd in all_responses.groupby('subject_id'):
        st_rts = subj_trial_responses[s] = {}
        for t, td in sd.groupby('trial_id'):
            if t not in trials: continue
            st_rts[t] = td.cross_time.values

    #dt = np.median(np.diff(all_trajectories['time_c']))
    dt = 1/30

    #param = {'std': 0.6970908298337709, 'damping': 0.46041074483528815, 'scale': 1.0, 'tau_threshold': 2.0542953634220624, 'act_threshold': 1.0}
#param = {'std': 0.6944196353260089, 'damping': 0.4747255460581086, 'scale': 0.7848596401338991, 'tau_threshold': 1.984151857035362, 'act_threshold': 1.0}

    loss = vdd_loss(list(trials.values()), dt, N=100)
    liks = []


    vdd_fit = fit_vdd(list(trials.values()), dt)
    print("VDD")
    print(vdd_fit)
    tdm_fit = fit_tdm(list(trials.values()), dt)
    print("TDM")
    print(tdm_fit)
    
def ecdf(vs):
    vs = np.sort(vs)
    x = [-np.inf]
    cdf = [0]
    for v in vs:
        if not np.isfinite(v): continue
        if v == x[-1]:
            cdf[-1] += 1
        if v > x[-1]:
            cdf.append(cdf[-1] + 1)
            x.append(v)

    x = np.array(x)
    cdf = np.array(cdf, dtype=float)
    cdf /= len(vs)
    return scipy.interpolate.interp1d(x, cdf, kind='previous', fill_value='extrapolate')

def plot_keio(country='uk'):
    dt = 1/30
    all_trajectories = pd.read_csv('d2p_trajectories.csv')
    all_responses = pd.read_csv(f'd2p_cross_times_{country}.csv')
    all_responses[["subject_id", "trial_number"]] = all_responses.unique_ID.str.split("_", 1, expand=True)
    responses = dict(list(all_responses.groupby('trial_id')))
    
    vddp = vddm_params[f'keio_{country}']
    tdmp = tdm_params[f'keio_{country}']
    
    #vddp = vddm_params['unified']
    #tdmp = tdm_params['unified']

    vddm = Vddm(dt=dt, **model_params(vddp))
    tdm = Tdm(**model_params(tdmp))

    #vddm = Vddm(dt=dt, **vddm_params[f'keio_{country}'])
    #tdm = Tdm(**tdm_params[f'keio_{country}'])
    
    pdf = PdfPages(f"keiofit_{country}.pdf")
    def show():
        pdf.savefig()
        plt.close()
    
    """
    all_trajectories = all_trajectories.query('trial_n > 2')
    tau_types = defaultdict(list)
    for trial, td in all_trajectories.groupby('trial_n'):
        has_decel = np.std(td.speed.values) > 0.01
        tau_type = trial
        if not has_decel:
            tau_type = round(td.distance.values[0]/td.speed.values[0], 1)
        
        tau_types[(tau_type, has_decel)].append((trial, td))
    """
    stats = []
    allpreds = []
    allpreds_tdm = []
    allcts = []
    trials = get_keio_trials(country=country)
    for traj, tr in trials:
            #plt.plot(td.time_c, td.distance)
            #ts = td.time_c.values
            #tr = responses[trial]
            #tau = td.distance/td.speed
            #td['ehmi'] = False
            ts = traj.time
            tau_vddm = mangle_tau(traj, **vddp)
            vddm_pdf = vddm.decisions(actgrid, tau_vddm)

            tau_tdm = mangle_tau(traj, **tdmp)
            tdm_pdf = tdm.decisions(tau_tdm, dt)

            stats.append(dict(
                mean=np.mean(tr),
                mean_vdd=np.dot(np.array(vddm_pdf.ps)*dt/(1 - vddm_pdf.uncrossed), ts),
                mean_tdm=np.dot(np.array(tdm_pdf.ps)*dt/(1 - tdm_pdf.uncrossed), ts),
                has_accel=np.std(traj.speed) > 0.01
                ))
            
            allcts.append(tr[np.isfinite(tr)])
            allpreds.append(
                    scipy.interpolate.interp1d(ts, np.array(vddm_pdf.ps)*len(allcts[-1]), bounds_error=False, fill_value=(0, 0))
                    )
            allpreds_tdm.append(
                    scipy.interpolate.interp1d(ts, np.array(tdm_pdf.ps)*len(allcts[-1]), bounds_error=False, fill_value=(0, 0))
                    )

            label = "Empirical v0 {td.speed[0]:.1f} m/s"
            plt.plot(ts, ecdf(tr)(ts)*100, label=f'Empirical, d0={traj.distance[0]:.1f} m')
            plt.title(f"Keio {country} trial type")
            plt.plot(ts, np.cumsum(np.array(vddm_pdf.ps)*dt)*100, 'k', label='VDDM')
            plt.plot(ts, np.cumsum(np.array(tdm_pdf.ps)*dt)*100, 'k--', label='TDM')
            plt.legend()
            plt.xlim(ts[0], ts[-1])
            plt.ylim(-1, 101)
            plt.ylabel('Percentage crossed')
            plt.xlabel('Time (seconds)')
            plt.twinx()
            plt.plot(ts, traj.distance/traj.speed, label='Time to arrival', color='black', alpha=0.5)
            plt.ylabel('Time to arrival (seconds)')
            plt.ylim(0, 8)
            show()

    stats = pd.DataFrame.from_records(stats)
    vstats = stats[~stats.has_accel]
    plt.plot(vstats['mean'], vstats.mean_vdd, 'C0o', label='VDDM (constant speed)')
    plt.plot(vstats['mean'], vstats.mean_tdm, 'C1o', label='TDM (constant speed)')
    
    vstats = stats[stats.has_accel]
    plt.plot(vstats['mean'], vstats.mean_vdd, 'C0x', label='VDDM (variable speed)')
    plt.plot(vstats['mean'], vstats.mean_tdm, 'C1x', label='TDM (variable speed)')
    
    rng = stats['mean'].min(), stats['mean'].max()
    plt.plot(rng, rng, 'k-', alpha=0.3)
    plt.legend()
    plt.xlabel('Measured mean crossing time (seconds)')
    plt.ylabel('Predicted mean crossing time (seconds)')
    plt.axis('equal')
    show()


    allcts = np.concatenate(allcts)
    allt = (np.concatenate([i.x for i in allpreds]))
    rng = np.arange(np.min(allcts), np.max(allcts), 0.05)
    pred = np.sum([i(rng) for i in allpreds], axis=0)/len(allcts)
    pred_tdm = np.sum([i(rng) for i in allpreds_tdm], axis=0)/len(allcts)
    plt.plot(rng, pred, color='black', label='VDDM')
    plt.plot(rng, pred_tdm, '--', color='black', label='TDM')

    rng = np.arange(rng[0], rng[-1], 0.2)
    plt.hist(allcts, bins=rng, density=True, color='C0', label='Observed')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Crossing probability density')
    show()

    pdf.close()

        
def plot_hiker():
    dt = 1/30
   
    data = pd.read_csv('hiker_cts.csv')
    data = data.query('braking_condition <= 3')
    
    pdf = PdfPages(f"hikerfit.pdf")
    def show():
        pdf.savefig()
        plt.close()
    
    leader_start = 100
    DT = 1/30
    trials = []
    trajectories = []
    #for i, (g, d) in enumerate(data.groupby(['time_gap', 'speed', 'is_braking', 'has_ehmi'])):
    """
    for i, (g, d) in enumerate(get_hiker_trials()):
        time_gap, speed, is_braking, has_hmi = g
        
        starttime = -leader_start/speed
        endtime = starttime + 20
        if not is_braking:
            endtime = time_gap

        crossing_times = d.crossing_time.values #- starttime
        crossing_times[~np.isfinite(crossing_times)] = np.inf
        #ts = np.arange(starttime, endtime, DT)
        
        traj, lead_traj = get_trajectory(time_gap, speed, is_braking, has_hmi)
        trajectories.append(dict(
            traj=traj, traj_lead=lead_traj,
            trial_id=i,
            #ts=ts, freetime=starttime, speed=lag_speed,
            initial_speed=speed,
            #tau_lag=tau_lag, tau_lead=tau_lead,
            crossing_times=crossing_times,
            #lag_distance=lag_x,
            #t_brake=t_brake, t_stop=t_stop,
            time_gap=time_gap, is_braking=is_braking, has_hmi=has_hmi))
    """
    vddm = Vddm(dt=dt, **model_params(vddm_params[f'hiker']))
    tdm = Tdm(**model_params(tdm_params[f'hiker']))
    stats = []
    
    def key(x):
        time_gap = round(x['time_gap'], 1)
        initial_speed = round(x['initial_speed'], 1)
        #if not x['is_braking']:
        #    initial_speed = -1 # Hack!
        return (x['is_braking'], initial_speed, time_gap, x['has_hmi'])

    allpreds = []
    allpreds_tdm = []
    allcts = []
    #for _, sametau in groupby(sorted(trajectories, key=key), key=key):
    #    for i, trial in enumerate(sametau):
    for traj, traj_lead, tr in get_hiker_trials():
        is_braking = np.std(traj['speed']) > 0.01
        initial_speed = traj_lead['speed'][0]
        time_gap = (traj['distance'] - traj_lead['distance'])[0]/initial_speed
        has_hmi = np.any(traj['ehmi'])

        g = is_braking, round(initial_speed, 1), round(time_gap, 1), has_hmi
        #plt.plot(td.time_c, td.distance)
        plt.title(g)
        #plt.title(f"HIKER v0 {trial['speed'][0]:.1f} m/s")
        ts = traj['time']
        #tr = trial['crossing_times']
        
        p = vddm_params['unified']
        vdd_taus = mangle_tau(traj, traj_lead, **p), mangle_tau(traj_lead, **p)
        vddm_pdf = vddm.blocker_decisions(actgrid, *vdd_taus)
        p = tdm_params['unified']
        tdm_taus = mangle_tau(traj, traj_lead, **p), mangle_tau(traj_lead, **p)
        tdm_pdf = tdm.blocker_decisions(*tdm_taus, dt)
        
        allcts.append(tr[np.isfinite(tr)])
        allpreds.append(
                scipy.interpolate.interp1d(ts, np.array(vddm_pdf.ps)*len(allcts[-1]), bounds_error=False, fill_value=(0, 0))
                )
        allpreds_tdm.append(
                scipy.interpolate.interp1d(ts, np.array(tdm_pdf.ps)*len(allcts[-1]), bounds_error=False, fill_value=(0, 0))
                )

        
        cdf = ecdf(tr)(ts)
        cdf_vdd = np.cumsum(np.array(vddm_pdf.ps)*dt)
        cdf_tdm = np.cumsum(np.array(tdm_pdf.ps)*dt)

        tau = traj['tau']
        tau_b = traj_lead['tau']
        #early_t = np.min(ts[(tau_b > 0) & (tau <= 0)], initial=np.inf)
        #early_t = min(trial['t_stop'], early_t, ts[-1])
        #early_i = np.searchsorted(ts, early_t)
        stats.append(dict(
            mean=np.mean(tr[np.isfinite(tr)]),
            mean_vdd=np.dot(np.array(vddm_pdf.ps)*dt/(1 - vddm_pdf.uncrossed), ts),
            mean_tdm=np.dot(np.array(tdm_pdf.ps)*dt/(1 - tdm_pdf.uncrossed), ts),
            has_accel=np.std(traj['speed']) > 0.01
            ))
        
        d0 = scipy.interpolate.interp1d(ts, traj['distance'])(0)
        plt.plot(ts, ecdf(tr)(ts)*100, label=f'Empirical, d0={d0:.1f} m, ehmi={has_hmi}')
        
        plt.plot(ts, np.cumsum(np.array(vddm_pdf.ps)*dt)*100, 'k', label='VDDM')
        plt.plot(ts, np.cumsum(np.array(tdm_pdf.ps)*dt)*100, 'k--', label='TDM')


        plt.xlim(-1, ts[-1])
        plt.ylim(-1, 101)
        plt.ylabel('Percentage crossed')
        plt.xlabel('Time (seconds)')
        plt.legend()
        plt.twinx()
        plt.ylabel('Time to arrival (seconds)')
        plt.plot(ts, tau, label='Time to arrival', color='black', alpha=0.5)
        plt.plot(ts, tau_b, 'k--', alpha=0.5)
        plt.ylim(0, 8)
        show()

    stats = pd.DataFrame.from_records(stats)
    rng = stats['mean'].min(), stats['mean'].max()
    plt.plot(rng, rng, 'C0-')
    
    plt.plot(stats['mean'], stats.mean_vdd, 'ko', label='VDDM')
    plt.plot(stats['mean'], stats.mean_tdm, 'kx', label='TDM')
    
    #vstats = stats[~stats.has_accel]
    #plt.plot(vstats['mean'], vstats.mean_vdd, 'C0o', label='VDDM (constant speed)')
    #plt.plot(vstats['mean'], vstats.mean_tdm, 'C1o', label='TDM (constant speed)')
    #vstats = stats[stats.has_accel]
    #plt.plot(vstats['mean'], vstats.mean_vdd, 'C0x', label='VDDM (variable speed)')
    #plt.plot(vstats['mean'], vstats.mean_tdm, 'C1x', label='TDM (variable speed)')
    
    plt.legend()
    plt.xlabel('Measured mean crossing time (seconds)')
    plt.ylabel('Predicted mean crossing time (seconds)')
    plt.axis('equal')
    show()
    
    allcts = np.concatenate(allcts)
    allt = (np.concatenate([i.x for i in allpreds]))
    rng = np.arange(np.min(allcts), np.max(allcts), 0.05)
    pred = np.sum([i(rng) for i in allpreds], axis=0)/len(allcts)
    pred_tdm = np.sum([i(rng) for i in allpreds_tdm], axis=0)/len(allcts)
    plt.plot(rng, pred, color='black', label='VDDM')
    plt.plot(rng, pred_tdm, '--', color='black', label='TDM')

    rng = np.arange(rng[0], rng[-1], 0.2)
    plt.hist(allcts, bins=rng, density=True, color='C0', label='Observed')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Crossing probability density')
    show()
    pdf.close()
    
DT = 1/30
leader_start = 100
vehicle_length = 5.0
def get_trajectory(time_gap, speed, is_braking, has_hmi, duration=20, end_at_passed=True, **kwargs):
    starttime = -(leader_start - vehicle_length)/speed
    endtime = starttime + duration
    if not is_braking and end_at_passed:
        endtime = time_gap

    ts = np.arange(starttime, endtime, DT)

    # Compensate for the car length?
    #distance = time_gap*speed
    #distance += vehicle_length
    #time_gap = distance/speed

    lag_x, lag_speed, (t_brake, t_stop) = hikersim.simulate_trajectory(ts, time_gap, speed, is_braking, **kwargs)

    tau_lag = -lag_x/lag_speed
    #tau_lag[~np.isfinite(tau_lag)] = 1e9
    # The leading car's coordinates were logged as the rear end, so
    # compensate to get the front
    lead_dist = leader_start - vehicle_length - (ts - starttime)*speed
    tau_lead = lead_dist/speed
    #tau_lead[~np.isfinite(tau_lead)] = 1e9
    
    lead_traj = np.rec.fromarrays(
            (ts, lead_dist, np.repeat(speed, len(ts)), tau_lead, np.gradient(tau_lead, DT), np.zeros(len(ts))),
            names="time,distance,speed,tau,tau_dot,ehmi")
    
    ehmi = np.zeros(len(ts))
    if has_hmi:
        ehmi[ts >= t_brake] = 1.0
    
    traj = np.rec.fromarrays((ts, -lag_x, lag_speed, tau_lag, np.gradient(tau_lag, DT), ehmi), names="time,distance,speed,tau,tau_dot,ehmi")

    return traj, lead_traj

def plot_schematic_old_wtf():
    dt = 1/30
    speed = 15.6464
    time_gap = 4
    ehmi = True
    is_braking = True

    traj, traj_lead = get_trajectory(time_gap, speed, is_braking, ehmi)
    
    traj = traj[traj.time >= 0]
    traj = traj[traj.time < 8]

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(12, 2)
    
    tauax = fig.add_subplot(gs[0:3,0])
    tauax.set_ylabel("TTA (s)")
    tauax.plot(traj.time, traj.tau, 'k')
    tauax.set_ylim(0, 10)
    tauax.get_xaxis().set_visible(False)
    
    taudotax = fig.add_subplot(gs[3:6,0])
    taudotax.set_ylabel("TTA change (s/s)")
    taudotax.plot(traj.time, traj.tau_dot, 'k')
    taudotax.set_ylim(-2, 2)
    taudotax.get_xaxis().set_visible(False)
    
    distax = fig.add_subplot(gs[6:9,0])
    distax.set_ylabel("Distance (m)")
    distax.plot(traj.time, traj.distance, 'k')
    distax.get_xaxis().set_visible(False)

    ehmiax = fig.add_subplot(gs[9:12,0])
    ehmiax.set_ylabel("eHMI active")
    ehmiax.plot(traj.time, traj.ehmi, 'k')
    ehmiax.set_xlabel("Time (seconds)")

   
    params = vddm_params['unified']
    model = Vddm(dt=dt, **model_params(params))
    inp = mangle_tau(traj, **params)
    
    allweights = np.zeros((len(traj), actgrid.N))
    weights = np.zeros(actgrid.N)
    
    weights[actgrid.bin(0)] = 1.0
    undecided = 1.0
    crossed = []
    for i in range(len(traj)):
        weights, decided = model.step(actgrid, inp[i], weights, 1.0)
        undecided -= undecided*decided
        crossed.append(1 - undecided)
        allweights[i] = weights
        allweights[i] *= undecided
    
    inpax = fig.add_subplot(gs[0:4,1])
    inpax.plot(traj.time, inp, 'k')
    inpax.set_ylim(0, 10)
    inpax.set_ylabel("Observation")
    inpax.get_xaxis().set_visible(False)

    actax = fig.add_subplot(gs[4:8,1])
    actax.set_ylabel("Activation")
    actax.get_xaxis().set_visible(False)
 
    actax.pcolormesh(traj.time, np.linspace(actgrid.low(), actgrid.high(), actgrid.N),
            allweights.T/actgrid.dx,
            vmax=0.5, cmap='jet')
    #actax.plot(traj.time, inp)
    actax.set_ylim(actgrid.low(), params['act_threshold'])
    
    crossax = fig.add_subplot(gs[8:12,1])
    crossax.plot(traj.time, crossed, 'k')
    crossax.set_ylabel("Decided")
    
    crossax.set_xlabel("Time (seconds)")

    plt.show()

dt = 1/30
def plot_schematic():
    dt = 1/30
    speed = 15.6464
    time_gap = 4
    ehmi = True
    is_braking = False
    traj, traj_lead = get_trajectory(time_gap, speed, is_braking, ehmi, end_at_passed=False)
    plot_traj_schematic(traj)

def plot_keio_schematics():
    trials = get_keio_trials(include_constants=False, include_decels=True)
    for i, (traj, resp) in enumerate(trials):
        plot_traj_schematic(traj, resp)
        plt.show()
        #plt.savefig(f"figs/keio_decel_sample_{i:02d}.png", dpi=300)

def plot_traj_schematic(traj, rts):
    traj = traj[traj.time >= 0]

    params = vddm_params['keio_uk']

    model = Vddm(dt=dt, **model_params(params))
    inp = mangle_tau(traj, **params)

    ndparams = params.copy()
    ndparams['dot_coeff'] = 0.0
    ndmodel = Vddm(dt=dt, **model_params(ndparams))
    ndinp = mangle_tau(traj, **ndparams)

    #fig = plt.figure(constrained_layout=True)
    fig, axs = plt.subplots(nrows=3, sharex=True)
    
    densax = axs[0]
    densax.set_xlim(0, 10)
    bins = np.arange(*traj.time[[0, -1]], 0.5)
    densax.plot(traj.time, ecdf(rts)(traj.time), color='black', label='Data')
    #_, _, histplot = densax.hist(rts, bins, color='black', density=True, alpha=0.25, label='Data')
    
    ps = np.array(model.decisions(actgrid, inp).ps)
    cdf = np.cumsum(ps*dt)
    densax.plot(traj.time, cdf, 'C0', label='Model')
    ndps = np.array(ndmodel.decisions(actgrid, ndinp).ps)
    ndcdf = np.cumsum(ndps*dt)
    densax.plot(traj.time, ndcdf, 'C1--', label=r'Model w/o $\dot\tau$')
    allweights = np.zeros((len(traj), actgrid.N))
    weights = np.zeros(actgrid.N)
    
    actax = axs[-1]
    weights[actgrid.bin(0)] = 1.0
    undecided = 1.0
    crossed = []

    for i in range(len(traj)):
        weights, decided = model.step(actgrid, inp[i], weights, 1.0)
        undecided -= undecided*decided
        crossed.append(1 - undecided)
        allweights[i] = weights
        allweights[i] *= undecided
    
    actax.pcolormesh(traj.time, np.linspace(actgrid.low(), actgrid.high(), actgrid.N),
            allweights.T/actgrid.dx,
            vmax=0.8, cmap='jet')
    actax.set_ylabel("Activation")
    actax.set_ylim(-1, params['act_threshold'])

    tauax = axs[1]
    
    tauax.plot(traj.time, traj.tau, color='C0', label='tau')
    tauax.set_ylim(0, 5.0)
    tauax.set_ylabel("Tau (s)", color='C0')
    
    distax = tauax.twinx()
    distax.plot(traj.time, traj.distance, 'k-', label='Distance')
    distax.set_ylim(0, 100)
    distax.set_ylabel("Distance (m)")

    axs[-1].set_xlabel("Time (seconds)")

    densax.legend()
    densax.set_ylabel("Share crossed")

def plot_traj_schematic_old(traj, resp):
    
    traj = traj[traj.time >= 0]
    #traj = traj[traj.time < 8]

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(12, 2)
    
    tauax = fig.add_subplot(gs[0:3,0])
    tauax.set_ylabel("TTA (s)")
    tauax.plot(traj.time, traj.tau, 'k')
    tauax.set_ylim(0, 10)
    tauax.get_xaxis().set_visible(False)
    
    taudotax = fig.add_subplot(gs[3:6,0])
    taudotax.set_ylabel("TTA change (s/s)")
    taudotax.plot(traj.time, traj.tau_dot, 'k')
    taudotax.set_ylim(-2, 2)
    taudotax.get_xaxis().set_visible(False)
    
    distax = fig.add_subplot(gs[6:9,0])
    distax.set_ylabel("Distance (m)")
    distax.plot(traj.time, traj.distance, 'k')
    distax.get_xaxis().set_visible(False)

    ehmiax = fig.add_subplot(gs[9:12,0])
    ehmiax.set_ylabel("eHMI active")
    ehmiax.plot(traj.time, traj.ehmi, 'k')
    ehmiax.set_xlabel("Time (seconds)")

   
    params = vddm_params['unified']
    model = Vddm(dt=dt, **model_params(params))
    inp = mangle_tau(traj, **params)
    
    tparams = tdm_params['unified']
    tdm = Tdm(**model_params(tparams))
    tinp = mangle_tau(traj, **tparams)
    
    allweights = np.zeros((len(traj), actgrid.N))
    weights = np.zeros(actgrid.N)
    
    weights[actgrid.bin(0)] = 1.0
    undecided = 1.0
    crossed = []
    for i in range(len(traj)):
        weights, decided = model.step(actgrid, inp[i], weights, 1.0)
        undecided -= undecided*decided
        crossed.append(1 - undecided)
        allweights[i] = weights
        allweights[i] *= undecided
    
    inpax = fig.add_subplot(gs[0:3,1])
    inpax.plot(traj.time, inp, 'k')
    inpax.plot(traj.time, tinp, 'k--')
    inpax.set_ylim(0, 10)
    inpax.set_ylabel("Observation")
    inpax.get_xaxis().set_visible(False)

    actax = fig.add_subplot(gs[3:6,1])
    actax.set_ylabel("Activation")
    actax.get_xaxis().set_visible(False)
 
    actax.pcolormesh(traj.time, np.linspace(actgrid.low(), actgrid.high(), actgrid.N),
            allweights.T/actgrid.dx,
            vmax=0.5, cmap='jet')
    #actax.plot(traj.time, inp)
    actax.set_ylim(actgrid.low(), params['act_threshold'])

    tdm_pdf = np.array(tdm.decisions(tinp, dt).ps)
    tdm_crossed = np.cumsum(tdm_pdf*dt)
    
    crossax = fig.add_subplot(gs[6:9,1])
    crossax.plot(traj.time, crossed, 'k')
    #crossax.plot(traj.time, tdm_crossed, 'k--')
    crossax.plot(traj.time, ecdf(resp)(traj.time))
    crossax.set_ylabel("Decided")
    
    crossax.set_xlabel("Time (seconds)")
    
    pdfax = fig.add_subplot(gs[9:12,1])
    pdfax.plot(traj.time, model.decisions(actgrid, inp).ps, 'k')
    #pdfax.plot(traj.time, tdm_pdf, 'k--')
    pdfax.hist(resp, density=True, bins=np.arange(*traj.time[[0, -1]], 0.5))
    pdfax.set_ylabel("Decision pdf")
    
    pdfax.set_xlabel("Time (seconds)")
    
    fig.align_labels()
    plt.show()

def plot_keio_consts():
    trials = get_keio_trials(include_decels=False)
    params = vddm_params['keio_uk']
    model = Vddm(dt=dt, **model_params(params))

    
    grps = lambda itr, key=None: (groupby(sorted(itr, key=key), key=key))
    
    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, constrained_layout=True)
    cfig, caxs = plt.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True, constrained_layout=True)
    speeds = sorted(np.unique([round(x[0].speed[0], 1) for x in trials]))
    
    for row, (v0, trials) in enumerate(grps(trials, lambda x: round(x[0].speed[0], 1))):
        for col, (tau0, trials) in enumerate(grps(trials, lambda x: round(x[0].tau[0], 1))):
            ax = axs[row, col]
            cax = caxs[col]
            if col == 0:
                ax.set_ylabel(f"v0 = {v0} m/s\nCrossing probability density")
            if row == 0:
                ax.set_title(f"tau0 = {tau0} s")
                cax.set_title(f"tau0 = {tau0} s")
            if row == nrows - 1:
                ax.set_xlabel("Time (s)")

            trial = list(trials)
            assert len(trial) == 1
            trial = trial[0]
            traj, rts = trial
            bins = np.arange(*traj.time[[0, -1]], 0.5)
            _, _, histplot = ax.hist(rts, bins, color='black', density=True, alpha=0.25, label='Data')
            inp = mangle_tau(traj, **params)
            pdf = np.array(model.decisions(actgrid, inp).ps)
            pdfplot, = ax.plot(traj.time, pdf, label='Model')

            dax = ax.twinx()
            distplot, = dax.plot(traj.time, traj.distance, color='black', label='Vehicle distance')
            dax.set_ylim(0, 100)

            if col != ncols - 1:
                dax.tick_params(labelright=False)
            else:
                dax.set_ylabel("Distance (m)")


            if col == ncols - 1 and row == 0:
                ax.legend(*zip(*((p, p.get_label()) for p in (histplot[0], pdfplot, distplot))))

            ax.set_xlim(0, 10)
            
            cdf = np.cumsum(pdf*DT)
            cax.set_ylim(-0.01, 1.01)
            cax.set_xlim(0, 10)
            cax.plot(traj.time, cdf, color=f"C{row}")
            
            cax.plot(traj.time, ecdf(rts)(traj.time), '--', color=f"C{row}")
            
            if not hasattr(cax, '_twinx'):
                cax._twinx = dax = cax.twinx()
                dax.set_ylim(0, 100)
            cax._twinx.plot(traj.time, traj.distance, '-', color=f"C{row}", alpha=0.5)
        
            if cax.is_first_row() and cax.is_first_col():
                handles = [
                    Line2D([0], [0], color='black', linestyle='solid', label='Model'),
                    Line2D([0], [0], color='black', linestyle='dashed', label='Data'),
                    Line2D([0], [0], color='black', linestyle='solid', alpha=0.5, label='Vehicle distance')
                        ]
                for si, speed in enumerate(speeds):
                    handles.append(Patch(facecolor=f'C{si}', label=f'Speed {speed} m/s')),
                cax.legend(handles=handles)
            
            if cax.is_first_col():
                cax.set_ylabel('Share passed')
            if cax.is_last_col():
                cax._twinx.set_ylabel("Distance (meters)")
            else:
                cax._twinx.get_yaxis().set_ticklabels([])
            if cax.is_last_row():
                cax.set_xlabel("Time (seconds)")

    plt.show()

def plot_hiker_consts():
    trials = get_hiker_trials(include_decels=False)
    params = vddm_params['hiker']
    model = Vddm(dt=dt, **model_params(params))

    
    grps = lambda itr, key=None: (groupby(sorted(itr, key=key), key=key))
    
    nrows = 3
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, constrained_layout=True)
    cfig, caxs = plt.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True, constrained_layout=True)
    
    speeds = sorted(np.unique([round(x[0].speed[0], 1) for x in trials]))
    
    for row, (v0, trials) in enumerate(grps(trials, lambda x: round(x[0].speed[0], 1))):
        for col, (tau0, trials) in enumerate(grps(trials, lambda x: round(x[0].tau[0] - x[1].tau[0], 1))):
            ax = axs[row, col]
            cax = caxs[col]
            if col == 0:
                ax.set_ylabel(f"v0 = {v0} m/s\nCrossing probability density")
                cax.set_ylabel(f"v0 = {v0} m/s\nShare crossed")
            if row == 0:
                ax.set_title(f"tau0 = {tau0} s")
                cax.set_title(f"tau0 = {tau0} s")
            if row == nrows - 1:
                ax.set_xlabel("Time (s)")

            trial = list(trials)
            assert len(trial) == 1
            trial = trial[0]
            traj, lead_traj, rts = trial
            bins = np.arange(*traj.time[[0, -1]], 0.25)
            
            hist, wtfbins = np.histogram(rts[np.isfinite(rts)], bins=bins, density=True)
            crosshare = np.sum(np.isfinite(rts))/len(rts)
            _, _, histplot = ax.hist(bins[:-1], bins=wtfbins, weights=hist*crosshare, color='black', alpha=0.25, label='Data')
            inp = mangle_tau(traj, lead_traj, **params)
            lead_inp = mangle_tau(lead_traj, **params)
            pdf = np.array(model.blocker_decisions(actgrid, inp, lead_inp).ps)
            pdfplot, = ax.plot(traj.time, pdf, label='Model')

            dax = ax.twinx()
            distplot, = dax.plot(traj.time, traj.distance, color='black', label='Vehicle distance')
            distplot, = dax.plot(traj.time, lead_traj.distance, '--', color='black', label='Lead vehicle distance')
            dax.set_ylim(0, 100)

            if col != ncols - 1:
                dax.tick_params(labelright=False)
            else:
                dax.set_ylabel("Distance (m)")


            #if col == ncols - 1 and row == 0:
            if col == 0 and row == 0:
                ax.legend(*zip(*((p, p.get_label()) for p in (histplot[0], pdfplot, distplot))))
                print(distplot.get_label())

            ax.set_xlim(-1, 3)
            
            cdf = np.cumsum(pdf*DT)
            cax.set_xlim(-1, 4)
            cax.set_ylim(-0.01, 1.01)
            if not hasattr(cax, '_twinx'):
                cax._twinx = dax = cax.twinx()
                dax.set_ylim(0, 100)
            
            cax._twinx.plot(traj.time, traj.distance, '-', color=f"C{row}", alpha=0.5)
 
            cax.plot(traj.time, cdf, color=f"C{row}")
            
            cax.plot(traj.time, ecdf(rts)(traj.time), '--', color=f"C{row}")

            if cax.is_first_row() and cax.is_first_col():
                handles = [
                    Line2D([0], [0], color='black', linestyle='solid', label='Model'),
                    Line2D([0], [0], color='black', linestyle='dashed', label='Data'),
                    Line2D([0], [0], color='black', linestyle='solid', alpha=0.5, label='Vehicle distance')
                        ]
                for si, speed in enumerate(speeds):
                    handles.append(Patch(facecolor=f'C{si}', label=f'Speed {speed} m/s')),
                cax.legend(handles=handles)
            
            if cax.is_first_col():
                cax.set_ylabel('Share passed')
            if cax.is_last_col():
                cax._twinx.set_ylabel("Distance (meters)")
            else:
                cax._twinx.get_yaxis().set_ticklabels([])
            if cax.is_last_row():
                cax.set_xlabel("Time (seconds)")
            

    plt.show()

def plot_hiker_decels():
    trials = get_hiker_trials(include_constants=False)
    params = vddm_params['hiker']
    model = Vddm(dt=dt, **model_params(params))

    
    grps = lambda itr, key=None: (groupby(sorted(itr, key=key), key=key))
    
    nrows = 3
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, constrained_layout=True)
    cfig, caxs = plt.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True, constrained_layout=True)
    
    for row, (v0, trials) in enumerate(grps(trials, lambda x: round(x[0].speed[0], 1))):
        for col, (tau0, trials) in enumerate(grps(trials, lambda x: round(x[0].tau[0] - x[1].tau[0], 1))):
            ax = axs[row, col]
            if col == 0:
                #ax.set_ylabel(f"v0 = {v0} m/s\nCrossing probability density")
                ax.set_ylabel(f"v0 = {v0} m/s\nShare passed")
            if row == 0:
                ax.set_title(f"tau0 = {tau0} s")
            if row == nrows - 1:
                ax.set_xlabel("Time (s)")
            
            for (has_ehmi, trials) in grps(trials, lambda x: np.any(x[0].ehmi != 0)):
                color = ['C0', 'C1'][has_ehmi]
                trial = list(trials)
                assert len(trial) == 1
                trial = trial[0]
                traj, lead_traj, rts = trial
                bins = np.arange(*traj.time[[0, -1]], 0.25)
                
                hist, wtfbins = np.histogram(rts[np.isfinite(rts)], bins=bins, density=True)
                crosshare = np.sum(np.isfinite(rts))/len(rts)
                #_, _, histplot = ax.hist(bins[:-1], bins=wtfbins, weights=hist*crosshare, color=color, alpha=0.25, label='Data')
                inp = mangle_tau(traj, lead_traj, **params)
                lead_inp = mangle_tau(lead_traj, **params)
                pdf = np.array(model.blocker_decisions(actgrid, inp, lead_inp).ps)
                #pdfplot, = ax.plot(traj.time, pdf, label='Model', color=color)
                cdf = np.cumsum(pdf*DT)
                ax.plot(traj.time, cdf, '-', color=color)
                ax.plot(traj.time, ecdf(rts)(traj.time), '--', color=color)


            dax = ax.twinx()
            distplot, = dax.plot(traj.time, traj.distance, color='black', alpha=0.5, label='Vehicle distance')
            distplot, = dax.plot(traj.time, lead_traj.distance, '--', color='black', alpha=0.5, label='Lead vehicle distance')
            dax.set_ylim(0, 100)

            if col != ncols - 1:
                dax.tick_params(labelright=False)
            else:
                dax.set_ylabel("Distance (m)")


            #if col == ncols - 1 and row == 0:
            #if col == 0 and row == 0:
            #    ax.legend(*zip(*((p, p.get_label()) for p in (histplot[0], pdfplot, distplot))))
            #    print(distplot.get_label())

            ax.set_xlim(-1, 10)
            
            cdf = np.cumsum(pdf*DT)
            cax = caxs[col]
            cax.plot(traj.time, cdf, color=f"C{row}")
            
            cax.plot(traj.time, ecdf(rts)(traj.time), '--', color=f"C{row}")
            

    plt.show()


def plot_keio_decels_old():
    import itertools
    trials = get_keio_trials(include_constants=False)
    params = vddm_params['keio_uk']
    model = Vddm(dt=dt, **model_params(params))

    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, constrained_layout=True)
    flataxs = list(itertools.chain(*axs))
    for i, (traj, rts) in enumerate(trials):
        tau0 = round(traj.tau[0], 1)
        v0 = round(traj.speed[0], 1)
        stopdist = round(np.min(traj.distance))
        ax = flataxs[i]
        ax.set_title(f"tau0 {tau0} s, v0 {v0} m/s, stopd {stopdist} m")
        bins = np.arange(*traj.time[[0, -1]], 0.5)
        _, _, histplot = ax.hist(rts, bins, color='black', density=True, alpha=0.25, label='Data')
        inp = mangle_tau(traj, **params)
        pdf = np.array(model.decisions(actgrid, inp).ps)
        pdfplot, = ax.plot(traj.time, pdf, label='Model')
        dax = ax.twinx()
        ax.twin = dax
        distplot, = dax.plot(traj.time, traj.distance, color='black', label='Vehicle distance')
        dax.set_ylim(0, 100)
        #print(traj.tau[0], traj.speed[0], np.min(traj.distance))

    flataxs[-1].legend(*zip(*((p, p.get_label()) for p in (histplot[0], pdfplot, distplot))))
    for ax in axs[-1]:
        ax.set_xlabel("Time (s)")

    for ax in axs[:,0]:
        ax.set_ylabel("Density")
    for ax in axs[:,-1]:
        if hasattr(ax, 'twin'):
            ax.twin.set_ylabel("Distance (m)")
        else:
            dax = ax.twinx()
            dax.set_ylabel("Distance (m)")
            dax.set_ylim(0, 100)
    plt.show()

def plot_keio_decels():
    import itertools
    trials = get_keio_trials(include_constants=False)
    params = vddm_params['unified']
    model = Vddm(dt=dt, **model_params(params))
    
    speeds = sorted(np.unique([round(x[0].speed[0], 1) for x in trials]))
    taus = sorted(np.unique([round(x[0].tau[0], 1) for x in trials]))
    stopdists = sorted(np.unique([round(np.min(x[0].distance), 1) for x in trials]))
    
    stopdcolors = dict(zip(stopdists, ('C0', 'C1')))
    
    grps = lambda itr, key=None: (groupby(sorted(itr, key=key), key=key))
    
    fig, axs = plt.subplots(nrows=len(speeds), ncols=len(taus), sharex=True, sharey=True, constrained_layout=True)
    
    for row, (v0, trials) in enumerate(grps(trials, lambda x: round(x[0].speed[0], 1))):
        for col, (tau0, trials) in enumerate(grps(trials, lambda x: round(x[0].tau[0], 1))):
            ax = axs[row, col]
            ax._twinx = ax.twinx()
            ax._twinx.set_ylim(0, 100)
            ax.set_xlim(0, 15)
            for stopd, trial in grps(trials, lambda x: round(np.min(x[0].distance), 1)):
                trial = list(trial)
                assert(len(trial) == 1)
                traj, rts = trial[0]
                color = stopdcolors[stopd]
                ax._twinx.plot(traj.time, traj.distance, alpha=0.5, color=color)
                
                bins = np.arange(*traj.time[[0, -1]], 0.5)
                inp = mangle_tau(traj, **params)
                pdf = np.array(model.decisions(actgrid, inp).ps)
                cdf = np.cumsum(pdf*DT)
                
                ax.plot(traj.time, cdf, color=color)
                ax.plot(traj.time, ecdf(rts)(traj.time), '--', color=color)

            if ax.is_first_row() and ax.is_first_col():
                handles = [
                    Line2D([0], [0], color='black', linestyle='solid', label='Model'),
                    Line2D([0], [0], color='black', linestyle='dashed', label='Data'),
                    Line2D([0], [0], color='black', linestyle='solid', alpha=0.5, label='Vehicle distance')
                        ]
                for stopd, color in stopdcolors.items():
                    handles.append(Patch(facecolor=color, label=f'Stop distance {stopd} m')),
                ax.legend(handles=handles)
            
            if ax.is_first_col():
                ax.set_ylabel(f'v0 {v0} m/s\nShare passed')
            if ax.is_first_row():
                ax.set_title(f'tau0 {tau0} s')
            if ax.is_last_col():
                ax._twinx.set_ylabel("Distance (meters)")
            else:
                ax._twinx.get_yaxis().set_ticklabels([])
            if ax.is_last_row():
                ax.set_xlabel("Time (seconds)")
    plt.show()
    
    """
    for i, (traj, rts) in enumerate(trials):
        tau0 = round(traj.tau[0], 1)
        v0 = round(traj.speed[0], 1)
        stopdist = round(np.min(traj.distance))
        ax = flataxs[i]
        ax.set_title(f"tau0 {tau0} s, v0 {v0} m/s, stopd {stopdist} m")
        bins = np.arange(*traj.time[[0, -1]], 0.5)
        _, _, histplot = ax.hist(rts, bins, color='black', density=True, alpha=0.25, label='Data')
        inp = mangle_tau(traj, **params)
        pdf = np.array(model.decisions(actgrid, inp).ps)
        pdfplot, = ax.plot(traj.time, pdf, label='Model')
        dax = ax.twinx()
        ax.twin = dax
        distplot, = dax.plot(traj.time, traj.distance, color='black', label='Vehicle distance')
        dax.set_ylim(0, 100)
        #print(traj.tau[0], traj.speed[0], np.min(traj.distance))

    flataxs[-1].legend(*zip(*((p, p.get_label()) for p in (histplot[0], pdfplot, distplot))))
    for ax in axs[-1]:
        ax.set_xlabel("Time (s)")

    for ax in axs[:,0]:
        ax.set_ylabel("Density")
    for ax in axs[:,-1]:
        if hasattr(ax, 'twin'):
            ax.twin.set_ylabel("Distance (m)")
        else:
            dax = ax.twinx()
            dax.set_ylabel("Distance (m)")
            dax.set_ylim(0, 100)
    plt.show()
    """

def plot_keio_means():
    trials = get_keio_trials()
    params = vddm_params['unified']
    model = Vddm(dt=dt, **model_params(params))
    
    datameans = []
    modelmeans = []
    for traj, rts in trials:
        datameans.append(np.mean(rts))
        inp = mangle_tau(traj, **params)
        pdf = np.array(model.decisions(actgrid, inp).ps)
        modelmeans.append(np.dot(traj.time, pdf/np.sum(pdf)))

    plt.plot([1.0, 5.0], [1.0, 5.0], 'k--', label="Identity")
    plt.plot(modelmeans, datameans, 'o', label="Trial mean")
    plt.xlabel("Predicted crossing time (s)")
    plt.ylabel("Measured crossing time (s)")
    plt.legend()
    plt.show()

def plot_hiker_means():
    trials = get_hiker_trials()
    params = vddm_params['hiker']
    model = Vddm(dt=dt, **model_params(params))
    
    datameans = []
    modelmeans = []
    is_braking = []
    has_ehmi = []
    for traj, trajb, rts in trials:
        inp = mangle_tau(traj, trajb, **params)
        inp_lead = mangle_tau(trajb, **params)

        pred = model.blocker_decisions(actgrid, inp, inp_lead)

        pdf = np.array(pred.ps)
        pdf /= np.sum(pdf)
        uncrossed = pred.uncrossed
        # For non-braking the late crossings are censored,
        # so let's input them as 10, which is beyond the maximum time
        # of the lengthiest non-braking trial
        censortime = 5
        
        uncrossed_share = np.sum(~np.isfinite(rts))/len(rts)
        
        is_braking.append(np.std(traj.speed) > 0.1)
        has_ehmi.append(np.any(traj.ehmi))
        datameans.append((1 - uncrossed)*np.mean(rts[np.isfinite(rts)]) + uncrossed_share*censortime)
        modelmeans.append((1 - uncrossed)*np.dot(traj.time, pdf) + uncrossed*censortime)
    
    datameans = np.array(datameans)
    modelmeans = np.array(modelmeans)
    is_braking = np.array(is_braking)
    has_ehmi = np.array(has_ehmi)

    ehmi_trials = has_ehmi & is_braking
    noehmi_trials = ~has_ehmi & is_braking

    plt.plot([1.0, 5.0], [1.0, 5.0], 'k--', label="Identity")
    plt.plot(modelmeans[noehmi_trials], datameans[noehmi_trials], 'o', label="Yielding, no eHMI")
    plt.plot(modelmeans[ehmi_trials], datameans[ehmi_trials], 'o', label="Yielding, with eHMI")
    plt.plot(modelmeans[~is_braking], datameans[~is_braking], 'o', label="Non-yielding")
    plt.xlabel("Predicted mean crossing time (s)")
    plt.ylabel("Measured mean crossing time (s)")
    plt.legend()
    plt.show()

def plot_hiker_time_savings():
    trials = get_hiker_trials(include_constants=False)
    params = vddm_params['hiker']
    model = Vddm(dt=dt, **model_params(params))

    
    grps = lambda itr, key=None: (groupby(sorted(itr, key=key), key=key))
    
    savings = []
    for row, (v0, trials) in enumerate(grps(trials, lambda x: round(x[0].speed[0], 1))):
        for col, (tau0, trials) in enumerate(grps(trials, lambda x: round(x[0].tau[0] - x[1].tau[0], 1))):
            means = {}

            for (has_ehmi, trial) in grps(trials, lambda x: np.any(x[0].ehmi != 0)):
                trial = list(trial)
                assert len(trial) == 1
                trial = trial[0]
                traj, lead_traj, rts = trial
                bins = np.arange(*traj.time[[0, -1]], 0.25)
                
                
                inp = mangle_tau(traj, lead_traj, **params)
                lead_inp = mangle_tau(lead_traj, **params)
                pdf = np.array(model.blocker_decisions(actgrid, inp, lead_inp).ps)

                pdf /= np.sum(pdf)
                pred_mean = np.dot(traj.time, pdf)
                data_mean = np.mean(rts[np.isfinite(rts)])
                means[has_ehmi] = np.array([pred_mean, data_mean])
            
            savings.append(means[False] - means[True])
            
    
    savings = np.array(savings)
    plt.plot(*savings.T, 'o', label="eHMI trial")
    plt.plot([-0.0, 2.0], [-0.0, 2.0], 'k--', label='Identity')
    plt.xlabel("Mean predicted pedestrian time savings (s)")
    plt.ylabel("Mean measured pedestrian time savings (s)")
    plt.legend()
    plt.axis('equal')
    plt.show()

def plot_hiker_time_savings2():
    trials = get_hiker_trials(include_constants=False)
    params = vddm_params['hiker']
    model = Vddm(dt=dt, **model_params(params))

    
    grps = lambda itr, key=None: (groupby(sorted(itr, key=key), key=key))
    
    taus = np.linspace(2, 5, 10)

    speeds = np.unique([round(x[0].speed[0], 1) for x in trials])
    for i, speed in enumerate(speeds):
        savings = []
        for tau in taus:
            traj, lead_traj = get_trajectory(tau, speed, True, False)
            inp = mangle_tau(traj, lead_traj, **params)
            lead_inp = mangle_tau(lead_traj, **params)
            pdf = np.array(model.blocker_decisions(actgrid, inp, lead_inp).ps)
            pdf /= np.sum(pdf)
            vanilla_mean = np.dot(traj.time, pdf)

            traj, lead_traj = get_trajectory(tau, speed, True, True)
            inp = mangle_tau(traj, lead_traj, **params)
            lead_inp = mangle_tau(lead_traj, **params)
            pdf = np.array(model.blocker_decisions(actgrid, inp, lead_inp).ps)
            pdf /= np.sum(pdf)
            ehmi_mean = np.dot(traj.time, pdf)
            
            savings.append(vanilla_mean - ehmi_mean)
        
        plt.plot(taus, savings, color=f"C{i}")



    savings = []
    for row, (v0, trials) in enumerate(grps(trials, lambda x: round(x[0].speed[0], 1))):
        for col, (tau0, trials) in enumerate(grps(trials, lambda x: round(x[0].tau[0] - x[1].tau[0], 1))):
            means = {}

            for (has_ehmi, trial) in grps(trials, lambda x: np.any(x[0].ehmi != 0)):
                trial = list(trial)
                assert len(trial) == 1
                trial = trial[0]
                traj, lead_traj, rts = trial
                bins = np.arange(*traj.time[[0, -1]], 0.25)
                
                
                inp = mangle_tau(traj, lead_traj, **params)
                lead_inp = mangle_tau(lead_traj, **params)
                pdf = np.array(model.blocker_decisions(actgrid, inp, lead_inp).ps)

                pdf /= np.sum(pdf)
                pred_mean = np.dot(traj.time, pdf)
                data_mean = np.mean(rts[np.isfinite(rts)])
                means[has_ehmi] = np.array([pred_mean, data_mean])
            
            saving = (means[False] - means[True])
            plt.plot(tau0, saving[1], 'o', color=f"C{row}")
            
    
    savings = np.array(savings)
    plt.ylabel("Mean eHMI pedestrian time saving (s)")
    plt.xlabel("Initial time gap (s)")
    plt.show()




def plot_sample_trials():
    dt = 1/30
    speed = 13.4
    time_gap = 4
    
    p = vddm_params['unified']
    vddm = Vddm(dt=dt, **model_params(p))
    def predict(traj, traj_lead):
        vdd_taus = mangle_tau(traj, **p), mangle_tau(traj_lead, **p)
        return np.array(vddm.blocker_decisions(actgrid, *vdd_taus).ps)
    
    """
    p = tdm_params['hiker']
    tdm = Tdm(**model_params(p))
    def predict(traj, traj_lead):
        tdm_taus = mangle_tau(traj, **p), mangle_tau(traj_lead, **p)
        return np.array(tdm.blocker_decisions(*tdm_taus, dt=dt).ps)
    """
    
    data = pd.read_csv('hiker_cts.csv')
    data = data.query('braking_condition <= 3')
    
    nd = data.query("not is_braking and time_gap == @time_gap")
    for i, (s, sd) in enumerate(nd.groupby('speed')):
        #cartime = vehicle_length/s
        cartime = 0
        traj, traj_b = get_trajectory(time_gap, s, False, False)
        emp = ecdf(sd.crossing_time)
        fit_pdf = predict(traj, traj_b)
        fit_cdf = np.cumsum(fit_pdf*dt)
        
        color = f"C{i}"
        d0 = s*time_gap
        plt.plot(traj.time + cartime, emp(traj.time), color=color, label=f"Initial distance {round(d0, 1)} m")
        #bins = np.arange(traj.time[0], traj.time[-1], 0.1)
        #plt.hist(sd.crossing_time + cartime, histtype='step', density=True, bins=bins)
        plt.plot(traj.time + cartime, fit_cdf, '--', color=color)
    
    plt.xlabel("Time since first car crossing (seconds)")
    plt.ylabel("Share crossed")
    plt.legend()
    plt.xlim(-1, 4)
    plt.show()
    
    nd = data.query("is_braking and time_gap == @time_gap and abs(speed - @speed) < 0.1")
    for i, (ehmi, sd) in enumerate(nd.groupby('has_ehmi')):
        cartime = 0
        traj, traj_b = get_trajectory(time_gap, speed, True, ehmi)
        emp = ecdf(sd.crossing_time)
        fit_pdf = predict(traj, traj_b)
        fit_cdf = np.cumsum(fit_pdf*dt)
        
        color = f"C{i}"
        label = ["Without eHMI", "With eHMI"][bool(ehmi)]
        plt.plot(traj.time + cartime, emp(traj.time), color=color, label=label)
        #bins = np.arange(traj.time[0], traj.time[-1], 0.1)
        #plt.hist(sd.crossing_time + cartime, histtype='step', density=True, bins=bins)
        plt.plot(traj.time + cartime, fit_cdf, '--', color=color)
    
    plt.xlabel("Time since first car crossing (seconds)")
    plt.ylabel("Share crossed")
    plt.legend()
    plt.xlim(-1, 12)
    plt.show()
 
def plot_hiker_schematics():
    trials = get_hiker_trials(include_constants=False)
    for traj, lead_traj, resp in trials:
        plot_hiker_schematic(traj, lead_traj, resp)


 
def plot_hiker_schematic(traj, lead_traj, rts):
    params = vddm_params['hiker']
    model = Vddm(dt=dt, **model_params(params))
    
    model_lead = Vddm(dt=dt, **{**model_params(params), 'tau_threshold': np.inf})
    model_lag = Vddm(dt=dt, **{**model_params(params)})
    inp = mangle_tau(traj, lead_traj, **{**params})

    inp_nohmi = mangle_tau(traj, lead_traj, **{**params, 'ehmi_coeff': 0.0})
    inp_lead = mangle_tau(lead_traj, **params)

    #fig = plt.figure(constrained_layout=True)
    fig, axs = plt.subplots(nrows=3, sharex=True)
    
    densax = axs[0]
    #densax.set_xlim(0, 8)
    bins = np.arange(*traj.time[[0, -1]], 0.25)
    decision_ps = np.array(model.blocker_decisions(actgrid, inp, inp_lead).ps)
    decision_ps_nohmi = np.array(model.blocker_decisions(actgrid, inp_nohmi, inp_lead).ps)
    rts = np.random.choice(traj.time, size=30, p=decision_ps/np.sum(decision_ps))
    _, _, histplot = densax.hist(rts, bins, color='black', density=True, alpha=0.25, label='Data')

    densax.plot(traj.time, decision_ps, 'C0', label='Model')
    if np.any(traj.ehmi != 0):
        densax.plot(traj.time, decision_ps_nohmi, 'C1--', label='Model w/o eHMI')
    allweights = np.zeros((len(traj), actgrid.N))
    weights = np.zeros(actgrid.N)
    
    actax = axs[-1]

    lead_ps = np.array(model_lead.decisions(actgrid, inp_lead).ps)
    lead_cdf = np.cumsum(lead_ps/np.sum(lead_ps))
    weights[actgrid.bin(0)] = 1.0
    undecided = 1.0
    crossed = []
    for i in range(len(traj)):
        weights, decided = model.step(actgrid, inp[i], weights, lead_cdf[i])
        undecided -= undecided*decided
        crossed.append(1 - undecided)
        allweights[i] = weights
        allweights[i] *= undecided

    ldensax = densax.twinx()
    ldensax.plot(traj.time, lead_cdf, 'C0--', label='Model lead CDF')
    ldensax.legend(loc='upper left')
    ldensax.set_ylim(0, 1)
    
    actax.pcolormesh(traj.time, np.linspace(actgrid.low(), actgrid.high(), actgrid.N),
            allweights.T/actgrid.dx,
            vmax=0.8, cmap='jet')
    actax.set_ylabel("Activation")
    actax.set_ylim(-1, params['act_threshold'])

    tauax = axs[1]

    tauax.plot(traj.time, traj.tau, color='C0', label='tau')
    tauax.plot(traj.time, lead_traj.tau, '--', color='C0', label='tau lead')
    tauax.set_ylim(0, 5.0)
    tauax.set_ylabel("Tau (s)", color='C0')
    distax = tauax.twinx()
    distax.plot(traj.time, traj.distance, 'k-', label='Distance')
    distax.plot(traj.time, lead_traj.distance, 'k--', label='Distance lead')
    distax.set_ylim(0, 100)
    distax.set_ylabel("Distance (m)")
    
    axs[-1].set_xlabel("Time (seconds)")
    axs[-1].set_xlim(-2, min(10, traj.time[-1]))
    densax.legend()
    densax.set_ylabel("Crossing density")
    plt.savefig("/tmp/plot.png", dpi=600)
    plt.show()

   

def plot_fake_hiker_schematic():
    time_gap = 2.0
    speed = 30*0.44704
    is_braking = True
    has_hmi = False
    np.random.seed(0)

    traj, lead_traj = get_trajectory(time_gap, speed, is_braking, has_hmi, x_stop=-2.5, x_brake=-38.5)

    params = vddm_params['unified']
    model = Vddm(dt=dt, **model_params(params))
    
    model_lead = Vddm(dt=dt, **{**model_params(params), 'tau_threshold': np.inf})
    model_lag = Vddm(dt=dt, **{**model_params(params)})
    inp = mangle_tau(traj, lead_traj, **{**params})
    inp_lead = mangle_tau(lead_traj, **params)

    #fig = plt.figure(constrained_layout=True)
    fig, axs = plt.subplots(nrows=3, sharex=True)
    
    densax = axs[0]
    #densax.set_xlim(0, 8)
    bins = np.arange(*traj.time[[0, -1]], 0.5)
    decision_ps = np.array(model.blocker_decisions(actgrid, inp, inp_lead).ps)
    rts = np.random.choice(traj.time, size=30, p=decision_ps/np.sum(decision_ps))
    _, _, histplot = densax.hist(rts, bins, color='black', density=True, alpha=0.25, label='Data')

    densax.plot(traj.time, decision_ps, 'C0', label='Model')
    allweights = np.zeros((len(traj), actgrid.N))
    weights = np.zeros(actgrid.N)
    
    actax = axs[-1]

    lead_ps = np.array(model_lead.decisions(actgrid, inp_lead).ps)
    lead_cdf = np.cumsum(lead_ps/np.sum(lead_ps))
    weights[actgrid.bin(0)] = 1.0
    undecided = 1.0
    crossed = []
    for i in range(len(traj)):
        weights, decided = model.step(actgrid, inp[i], weights, lead_cdf[i])
        undecided -= undecided*decided
        crossed.append(1 - undecided)
        allweights[i] = weights
        allweights[i] *= undecided
    
    actax.pcolormesh(traj.time, np.linspace(actgrid.low(), actgrid.high(), actgrid.N),
            allweights.T/actgrid.dx,
            vmax=0.8, cmap='jet')
    actax.set_ylabel("Activation")
    actax.set_ylim(-1, params['act_threshold'])

    tauax = axs[1]

    tauax.plot(traj.time, traj.tau, color='C0', label='tau')
    tauax.set_ylim(0, 5.0)
    tauax.set_ylabel("Tau (s)", color='C0')
    distax = tauax.twinx()
    distax.plot(traj.time, traj.distance, 'k-', label='Distance')
    distax.set_ylim(0, 100)
    distax.set_ylabel("Distance (m)")
    
    axs[-1].set_xlabel("Time (seconds)")
    axs[-1].set_xlim(-1, 8)
    densax.legend()
    densax.set_ylabel("Crossing density")
    plt.savefig("/tmp/plot.png", dpi=600)
    plt.show()



if __name__ == '__main__':
    #plot_hiker()
    #plot_keio('uk')
    #plot_keio('japan')
    #print("keiouk")
    #fit_keio('uk')
    #print("keiojapan")
    #fit_keio('japan')
    #print("hiker")
    #fit_hiker()

    #plot_sample_trials()
    #plot_schematic()
    #plot_keio_schematics()
    fit_hiker_and_keio()

    #plot_hiker_schematics()
    #plot_hiker_means()
    #plot_hiker_time_savings()
    #plot_hiker_time_savings2()
    #plot_hiker_consts()
    #plot_hiker_decels()

    #plot_keio_consts()
    #plot_keio_decels()
    #plot_keio_means()
