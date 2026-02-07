"""
Method configuration factory.

Maps method name strings to configured optical flow objects,
reproducing load_of_method.m from the MATLAB codebase.
"""
from optical_flow.robust.robust_function import RobustFunction


def load_of_method(method):
    """Load a pre-configured optical flow method by name.

    Available methods:
        - 'classic+nl-fast': Classic+NL with reduced iterations (fast)
        - 'classic+nl': Classic+NL with texture decomposition & weighted median
        - 'classic+nl-full': Classic+NL with full weighted median version
        - 'hs-brightness': Horn-Schunck with brightness constancy
        - 'hs': Horn-Schunck with ROF texture constancy
        - 'ba-brightness': Black-Anandan with brightness constancy
        - 'ba', 'classic-l': BA with texture, lorentzian penalties
        - 'classic-c-brightness': Classic with charbonnier, brightness
        - 'classic-c': Classic with charbonnier, texture
        - 'classic++': Classic++ with gen. charbonnier, texture, bi-cubic
        - 'classic-c-a': Alt BA with charbonnier penalties

    Args:
        method: Method name string.

    Returns:
        ope: Configured optical flow object.
    """
    median_filter_size = [5, 5]

    if method == 'classic+nl-fast':
        ope = load_of_method('classic+nl')
        ope.max_iters = 3
        ope.gnc_iters = 2
        ope.display = True
        return ope

    elif method == 'classic+nl':
        from optical_flow.methods.classic_nl import ClassicNLOpticalFlow
        import numpy as np

        ope = ClassicNLOpticalFlow()
        ope.texture = True
        ope.median_filter_size = median_filter_size
        ope.alp = 0.95
        ope.area_hsz = 7
        ope.sigma_i = 7
        ope.color_images = np.ones((1, 1, 3))
        ope.lambda_ = 3
        ope.lambda_q = 3
        return ope

    elif method == 'classic+nl-full':
        ope = load_of_method('classic+nl')
        ope.fullVersion = True
        return ope

    elif method == 'hs-brightness':
        from optical_flow.methods.hs import HSOpticalFlow
        ope = HSOpticalFlow()
        ope.median_filter_size = median_filter_size
        ope.lambda_ = 10
        ope.lambda_q = 10
        return ope

    elif method == 'hs':
        from optical_flow.methods.hs import HSOpticalFlow
        ope = HSOpticalFlow()
        ope.median_filter_size = median_filter_size
        ope.texture = True
        ope.lambda_ = 40
        ope.lambda_q = 40
        ope.display = True
        return ope

    elif method == 'ba-brightness':
        from optical_flow.methods.ba import BAOpticalFlow
        ope = BAOpticalFlow()
        ope.median_filter_size = median_filter_size

        m = 'lorentzian'
        ope.spatial_filters = [__import__('numpy').array([[1, -1]]),
                               __import__('numpy').array([[1], [-1]])]
        ope.rho_spatial_u = [RobustFunction(m, 0.1), RobustFunction(m, 0.1)]
        ope.rho_spatial_v = [RobustFunction(m, 0.1), RobustFunction(m, 0.1)]
        ope.rho_data = RobustFunction(m, 3.5)
        ope.lambda_ = 0.045
        ope.lambda_q = 0.045
        return ope

    elif method in ('classic-l', 'ba'):
        ope = load_of_method('ba-brightness')
        ope.median_filter_size = median_filter_size
        ope.texture = True

        m = 'lorentzian'
        import numpy as np
        ope.spatial_filters = [np.array([[1, -1]]), np.array([[1], [-1]])]
        ope.rho_spatial_u = [RobustFunction(m, 0.03), RobustFunction(m, 0.03)]
        ope.rho_spatial_v = [RobustFunction(m, 0.03), RobustFunction(m, 0.03)]
        ope.rho_data = RobustFunction(m, 1.5)
        ope.lambda_ = 0.06
        ope.lambda_q = 0.06
        return ope

    elif method == 'classic-c-a':
        from optical_flow.methods.alt_ba import AltBAOpticalFlow
        import numpy as np

        ope = AltBAOpticalFlow()
        ope.median_filter_size = median_filter_size
        ope.texture = True

        m = 'charbonnier'
        ope.spatial_filters = [np.array([[1, -1]]), np.array([[1], [-1]])]
        ope.rho_spatial_u = [RobustFunction(m, 1e-3), RobustFunction(m, 1e-3)]
        ope.rho_spatial_v = [RobustFunction(m, 1e-3), RobustFunction(m, 1e-3)]
        ope.rho_data = RobustFunction(m, 1e-3)
        ope.display = False
        ope.lambda2 = 1e2
        ope.lambda3 = 1
        ope.weightRatio = ope.lambda2 / ope.lambda3
        ope.itersLO = 5
        ope.lambda_ = 5
        ope.lambda_q = 5
        return ope

    elif method == 'classic-c-brightness':
        from optical_flow.methods.ba import BAOpticalFlow
        import numpy as np

        ope = BAOpticalFlow()
        ope.median_filter_size = median_filter_size
        ope.texture = False

        m = 'charbonnier'
        ope.spatial_filters = [np.array([[1, -1]]), np.array([[1], [-1]])]
        ope.rho_spatial_u = [RobustFunction(m, 1e-3), RobustFunction(m, 1e-3)]
        ope.rho_spatial_v = [RobustFunction(m, 1e-3), RobustFunction(m, 1e-3)]
        ope.rho_data = RobustFunction(m, 1e-3)
        ope.lambda_ = 3
        ope.lambda_q = 3
        return ope

    elif method == 'classic-c':
        ope = load_of_method('classic-c-brightness')
        ope.texture = True
        ope.lambda_ = 5
        ope.lambda_q = 5
        return ope

    elif method == 'classic++':
        from optical_flow.methods.ba import BAOpticalFlow
        import numpy as np

        ope = BAOpticalFlow()
        ope.median_filter_size = median_filter_size
        ope.texture = True
        ope.interpolation_method = 'bi-cubic'

        m = 'generalized_charbonnier'
        a = 0.45
        sig = 1e-3
        ope.spatial_filters = [np.array([[1, -1]]), np.array([[1], [-1]])]
        ope.rho_spatial_u = [RobustFunction(m, sig, a), RobustFunction(m, sig, a)]
        ope.rho_spatial_v = [RobustFunction(m, sig, a), RobustFunction(m, sig, a)]
        ope.rho_data = RobustFunction(m, sig, a)
        ope.lambda_ = 3
        ope.lambda_q = 3
        return ope

    else:
        raise ValueError(f"Unknown optical flow method: '{method}'")
