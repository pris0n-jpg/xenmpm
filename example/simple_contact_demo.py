# -*- coding: utf-8 -*-
"""
简易接触场景使用示例
演示物体与传感器的接触仿真
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from xengym.render.simpleScene import SimpleContactScene, create_simple_scene


def demo_basic_contact():
    """基础接触演示"""
    print("🎯 === 基础接触演示 ===")
    
    # 创建简易场景
    scene = create_simple_scene(
        initial_height=25.0,  # 初始高度25mm
        visible=True
    )
    
    try:
        # 运行自动演示
        scene.run_demo(steps=40, step_size=0.6)
        
    finally:
        scene.close()


def demo_manual_control():
    """手动控制演示"""
    print("\n🎮 === 手动控制演示 ===")
    
    scene = SimpleContactScene(
        initial_height=30.0,
        visible=True,
        title="手动控制演示"
    )
    
    try:
        print("手动控制物体移动:")
        
        # 1. 设置物体位置
        print("\n1️⃣ 设置物体到不同位置...")
        positions = [
            [5.0, 0.0, 25.0],   # 右侧
            [-5.0, 0.0, 20.0],  # 左侧
            [0.0, 5.0, 15.0],   # 前方
            [0.0, 0.0, 10.0],   # 中心
        ]
        
        for i, pos in enumerate(positions):
            scene.set_object_pose(position=pos)
            result = scene.step()
            sensor_data = result['sensor_data']
            
            print(f"位置 {i+1}: {pos} -> "
                  f"接触={'是' if sensor_data['contact'] else '否'}, "
                  f"深度={sensor_data['depth']:.2f}mm")
            
            import time
            time.sleep(1.0)
        
        # 2. 逐步下降测试
        print("\n2️⃣ 逐步下降测试...")
        scene.reset()  # 重置到初始位置
        
        for step in range(20):
            result = scene.step({'move_down': 1.0})
            sensor_data = result['sensor_data']
            
            if sensor_data['contact']:
                print(f"步骤 {step+1}: 发生接触! "
                      f"深度={sensor_data['depth']:.2f}mm, "
                      f"力={sensor_data['force'][2]:.3f}N")
            
            time.sleep(0.2)
        
        # 3. 旋转测试
        print("\n3️⃣ 旋转测试...")
        scene.reset()
        scene.set_object_pose(position=[0, 0, 15])  # 设置到接触位置
        
        rotations = [
            [15, 0, 0],    # X轴旋转
            [0, 15, 0],    # Y轴旋转
            [0, 0, 15],    # Z轴旋转
            [10, 10, 10],  # 组合旋转
        ]
        
        for i, rot in enumerate(rotations):
            scene.set_object_pose(rotation=rot)
            result = scene.step()
            sensor_data = result['sensor_data']
            
            print(f"旋转 {i+1}: {rot}° -> "
                  f"接触={'是' if sensor_data['contact'] else '否'}, "
                  f"深度={sensor_data['depth']:.2f}mm")
            
            time.sleep(1.0)
        
        print("\n✅ 手动控制演示完成")
        
    finally:
        scene.close()


def demo_data_collection():
    """数据收集演示"""
    print("\n📊 === 数据收集演示 ===")
    
    scene = SimpleContactScene(
        initial_height=20.0,
        visible=False,  # 无可视化，专注数据收集
        title="数据收集"
    )
    
    # 收集接触数据
    contact_data = []
    
    print("收集接触数据...")
    for step in range(50):
        # 物体下降
        result = scene.step({'move_down': 0.3})
        
        # 记录数据
        contact_data.append({
            'step': step,
            'position': result['object_position'].copy(),
            'contact': result['sensor_data']['contact'],
            'depth': result['sensor_data']['depth'],
            'force': result['sensor_data']['force'].copy(),
            'timestamp': result['timestamp']
        })
        
        # 显示进度
        if step % 10 == 0:
            sensor_data = result['sensor_data']
            print(f"步骤 {step}: "
                  f"高度={result['object_position'][2]:.1f}mm, "
                  f"接触={'是' if sensor_data['contact'] else '否'}")
    
    scene.close()
    
    # 分析数据
    print(f"\n📈 数据分析:")
    contact_steps = [d for d in contact_data if d['contact']]
    
    print(f"总步数: {len(contact_data)}")
    print(f"接触步数: {len(contact_steps)}")
    print(f"接触率: {len(contact_steps)/len(contact_data)*100:.1f}%")
    
    if contact_steps:
        depths = [d['depth'] for d in contact_steps]
        forces = [d['force'][2] for d in contact_steps]
        
        print(f"最大接触深度: {max(depths):.2f}mm")
        print(f"平均接触深度: {np.mean(depths):.2f}mm")
        print(f"最大接触力: {abs(min(forces)):.3f}N")
        print(f"平均接触力: {abs(np.mean(forces)):.3f}N")
    
    return contact_data


def demo_different_objects():
    """不同物体演示"""
    print("\n🎲 === 不同物体演示 ===")
    
    # 测试不同的物体文件
    object_files = [
        "assets/obj/cube_15mm.obj",      # 立方体
        "assets/obj/circle_r4.STL",      # 圆形
        "assets/obj/handle.STL",         # 手柄
    ]
    
    print("将为每个物体创建独立场景进行演示")
    print("🔄 自动关闭旧窗口并创建新窗口\n")
    
    scene = None  # 保持场景引用
    
    for i, obj_file in enumerate(object_files):
        print(f"\n测试物体 {i+1}: {Path(obj_file).name}")
        
        try:
            # 如果有旧场景，先强制关闭和销毁
            if scene is not None:
                print("🔄 正在关闭旧场景...")
                scene.close()
                
                # 强制删除引用并触发垃圾回收
                del scene
                import gc
                gc.collect()
                
                # 短暂延迟确保资源释放
                import time
                time.sleep(1.0)
                print("✅ 旧场景已关闭")
            
            # 创建新场景
            print(f"🆕 创建新场景: {Path(obj_file).name}")
            scene = SimpleContactScene(
                object_file=obj_file,
                initial_height=25.0,
                visible=True,
                title=f"物体测试 - {Path(obj_file).name}"
            )
            
            # 快速演示
            scene.run_demo(steps=20, step_size=1.0)
            
            print(f"✅ 物体 {Path(obj_file).name} 演示完成")
            
            # 简单的暂停，让用户观察结果
            import time
            time.sleep(2.0)  # 暂停2秒让用户观察
            
            # 如果不是最后一个，询问用户是否继续
            if i < len(object_files) - 1:
                print("💡 按 Enter 继续下一个演示，或 Ctrl+C 退出...")
                try:
                    input()  # 等待用户按Enter
                except KeyboardInterrupt:
                    print("\n用户选择退出演示")
                    break
            
        except Exception as e:
            print(f"⚠ 物体 {obj_file} 测试失败: {e}")
    
    # 最后关闭场景
    if scene is not None:
        print("🔄 关闭最后一个场景...")
        scene.close()
        del scene
        import gc
        gc.collect()
    
    print("🎉 所有物体演示完成!")


if __name__ == '__main__':
    print("🎬 === 简易接触场景完整演示 ===")
    
    demos = [
        ("基础接触演示", demo_basic_contact),
        ("手动控制演示", demo_manual_control),
        ("数据收集演示", demo_data_collection),
        ("不同物体演示", demo_different_objects),
    ]
    
    try:
        for i, (name, demo_func) in enumerate(demos):
            print(f"\n{'='*50}")
            print(f"🎯 开始 {name} ({i+1}/{len(demos)})")
            print(f"{'='*50}")
            
            try:
                if demo_func == demo_data_collection:
                    contact_data = demo_func()
                else:
                    demo_func()
                    
                print(f"✅ {name} 完成")
                
                # 如果不是最后一个演示，询问是否继续
                if i < len(demos) - 1:
                    print(f"\n💡 按 Enter 继续下一个演示 ({demos[i+1][0]})，或 Ctrl+C 退出...")
                    input()
                    
            except KeyboardInterrupt:
                print(f"\n⏹ 用户中断 {name}")
                break
            except Exception as e:
                print(f"\n❌ {name} 过程中出错: {e}")
                import traceback
                traceback.print_exc()
                
                print("💡 按 Enter 继续下一个演示，或 Ctrl+C 退出...")
                try:
                    input()
                except KeyboardInterrupt:
                    break
        
        print("\n🎉 所有演示完成!")
        
    except KeyboardInterrupt:
        print("\n⏹ 用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc() 