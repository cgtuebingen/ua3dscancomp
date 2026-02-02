import torch

def sample_random_coordinate_system_vmf(kappa: float, sample_2d: torch.Tensor) -> torch.Tensor:
    ## sample 2d circle for x/y
    phi = torch.pi*2.0*sample_2d[0]
    x, y = torch.cos(phi), torch.sin(phi)

    xi = sample_2d[1]
    if kappa < 1.0e-6:
       # uniform sphere
       z = 1.0-2.0*sample_2d[1]

    elif kappa > 45.0:
        ## simplified sampling, as exp(2-kappa) goes to 0
        z = 1.0+torch.log1p(-xi)/kappa

    else:
        ## sample vMF for z (based on Jakob 2012)
        z = 1.0+torch.log1p(torch.exp(torch.as_tensor(-2.0*kappa))*xi-xi)/kappa

    # x*x+y*y+z*z = 1 <=> x*x+y*y = 1-z*z
    xy_scale = torch.sqrt(1.0-z*z)
    z_vec = torch.stack([x*xy_scale, y*xy_scale, z])

    # make a full coordinate system
    x_init = torch.stack([torch.ones_like(z), torch.zeros_like(z), torch.zeros_like(z)])
    y_vec = torch.nn.functional.normalize(torch.cross(z_vec, x_init), dim=0)
    x_vec = -torch.cross(z_vec, y_vec)

    coord_frame = torch.stack([x_vec, y_vec, z_vec], dim=1)

    # breakpoint()

    return coord_frame

def kappa_to_mean_cosine(kappa: float) -> float:
    kappa = torch.as_tensor(kappa)
    mean_cosine = (kappa-torch.tanh(kappa))/(kappa*torch.tanh(kappa))
    return mean_cosine.item()

def mean_cosine_to_kappa(mean_cos: float) -> float:
    # approximation by Banerjee et al. 2005
    r = mean_cos
    d = 3
    kappa = (r*d - (r*r*r)) / (1.0 - r*r)

    for i in range(1):
        # refine kappa using Newton-Raphson (see Footnote 3 in Banerjee et al. 2005)
        # approximation error for 1 iteration: less than 0.5% too small for 0 < kappa < 10
        mean_cos_estimate = kappa_to_mean_cosine(kappa)
        offset = mean_cos_estimate-mean_cos
        offset_gradient = 1.0-mean_cos_estimate*mean_cos_estimate-2.0*mean_cos/kappa
        step = offset/offset_gradient

        if not torch.isnan(torch.as_tensor(step)):
            kappa = kappa-step

    return kappa

def mean_angle_deg_to_kappa(angle: float) -> float:
    return mean_cosine_to_kappa(torch.cos(torch.as_tensor(angle*torch.pi/180.0)).item())

def test():
    kappa = float('inf')
    sample_2d = torch.rand((2,))
    coord_frame = sample_random_coordinate_system_vmf(kappa, sample_2d)
    assert(torch.allclose(coord_frame, torch.eye(3)))
    print('kappa', kappa)
    print('sample_2d', sample_2d)
    print('coord_frame', coord_frame)

    for angle in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        kappa = mean_angle_deg_to_kappa(angle)
        cos_sampled = []
        print()
        print('angle', angle)
        print('kappa', kappa)
        for i in range(1000):
            sample_2d = torch.rand((2,))
            coord_frame = sample_random_coordinate_system_vmf(kappa, sample_2d)
            cos_sampled.append(coord_frame[2,2])
            try:
                assert(torch.allclose(coord_frame[0].dot(coord_frame[1]), torch.as_tensor(0.0), atol=1.0e-6))
                assert(torch.allclose(coord_frame[0].dot(coord_frame[2]), torch.as_tensor(0.0), atol=1.0e-6))
                assert(torch.allclose(coord_frame[1].dot(coord_frame[2]), torch.as_tensor(0.0), atol=1.0e-6))
                assert(torch.allclose(torch.linalg.vector_norm(coord_frame, dim=0), torch.as_tensor(1.0), atol=1.0e-6))
                assert(torch.allclose(torch.linalg.vector_norm(coord_frame, dim=1), torch.as_tensor(1.0), atol=1.0e-6))
            except:
                breakpoint()
        print('mean angle:', torch.acos(torch.mean(torch.as_tensor(cos_sampled)))*180.0/torch.pi)

if __name__ == '__main__':
    test()



