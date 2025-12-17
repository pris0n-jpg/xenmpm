import argparse

from xengym import main, PROJ_DIR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Xense Sim')
    # 添加参数
    parser.add_argument('-f', '--fem_file', type=str, help='Path to the FEM file (default: %(default)s)', default=str(PROJ_DIR/"assets/data/fem_data_gel_2035.npz"))
    parser.add_argument('-u', '--urdf_file', type=str, help='Path to the URDF file (default: %(default)s)', default=str(PROJ_DIR/"assets/panda/panda_with_vectouch.urdf"))
    parser.add_argument('-o', '--object_file', type=str, help='Path to the object file (default: %(default)s)', default=str(PROJ_DIR/"assets/obj/letter.STL"))
    parser.add_argument('-l', '--show_left', help='Show left sensor (default: %(default)s)', action='store_true', default=False)
    parser.add_argument('-r', '--show_right', help='Show right sensor (default: %(default)s)', action='store_true', default=False)
    args = parser.parse_args()

    main(args)