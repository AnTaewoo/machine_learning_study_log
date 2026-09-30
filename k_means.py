import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_distance(p1_x, p1_y, p2_x, p2_y):
    d_x = (p1_x - p2_x) ** 2
    d_y = (p1_y - p2_y) ** 2

    return (d_x + d_y) ** 0.5


class Kmean:
    def __init__(self, points, k_size):
        self.input_size = len(points)
        self.k_size = k_size
        self.table = [[] for _ in range(k_size)]

        arr_index = 0
        for i in range(self.k_size):
            start_arr_index = arr_index
            current_cluster_size = self.input_size // self.k_size + (
                1 if i < self.input_size % self.k_size else 0
            )
            arr_index += current_cluster_size

            self.table[i] = points[start_arr_index:arr_index]

        self.centers = [{"x": 0, "y": 0} for _ in range(k_size)]

    def set_centers(self):
        _centers = [{"x": 0, "y": 0} for _ in range(self.k_size)]
        _table = self.table
        for i, points in enumerate(_table):
            sum_x = 0
            sum_y = 0
            for point in points:
                sum_x += point["x"]
                sum_y += point["y"]

            if len(points) != 0:
                center_x = sum_x / len(points)
                center_y = sum_y / len(points)
                _centers[i] = {"x": center_x, "y": center_y}

        for i, points in enumerate(_table):
            if len(points) == 0:
                del _table[i]
                del _centers[i]
                self.k_size -= 1

        self.table = _table
        self.centers = _centers
        return _centers

    def run_iteration(self):
        centers = self.set_centers()

        _table = [[] for _ in range(self.k_size)]
        for points in self.table:
            for point in points:
                distance_for_centers = [0 for _ in range(self.k_size)]
                for i, center in enumerate(centers):
                    distance = get_distance(
                        center["x"], center["y"], point["x"], point["y"]
                    )
                    distance_for_centers[i] = distance
                min_center = distance_for_centers.index(min(distance_for_centers))
                _table[min_center].append(point)

        self.table = _table
        return _table


if __name__ == "__main__":
    points = [
        {"x": -5, "y": 8},
        {"x": 10, "y": -3},
        {"x": -2, "y": -9},
        {"x": 4, "y": 1},
        {"x": -10, "y": 5},
        {"x": 3, "y": -2},
        {"x": -4, "y": 9},
        {"x": 11, "y": -6},
        {"x": -7, "y": 2},
        {"x": 5, "y": -8},
        {"x": -1, "y": 4},
        {"x": 8, "y": -3},
        {"x": -9, "y": -5},
        {"x": 2, "y": 7},
        {"x": -3, "y": 1},
        {"x": 12, "y": -2},
        {"x": 6, "y": 6},
        {"x": -8, "y": 3},
        {"x": 0, "y": -9},
        {"x": 4, "y": -1},
        {"x": -6, "y": -7},
        {"x": 13, "y": 5},
        {"x": -2, "y": 8},
        {"x": 7, "y": -4},
        {"x": -5, "y": -2},
        {"x": 10, "y": 0},
    ]
    k_size = 8
    loss = 1e-4
    model = Kmean(points, k_size)

    done = False
    prev_center = model.centers
    while not done:
        model.run_iteration()
        done = True

        centers = model.centers
        for k in range(model.k_size):
            distance = get_distance(
                prev_center[k]["x"],
                prev_center[k]["y"],
                centers[k]["x"],
                centers[k]["y"],
            )

            if distance > loss:
                done = False
                break
        prev_center = model.centers
        if done:
            print(
                f"{k_size} cluster에서 불필요한 cluster {k_size - model.k_size}개 제거 후, cluster_circles.png 가 생성되었습니다."
            )

    # ---------------------------------------------------
    # 데이터 시각화는 claude로 만들어서 처리했습니다~
    clusters = model.table
    centers = model.centers

    plt.figure(figsize=(8, 8))
    colors = [
        "#3498db",  # 1. 블루
        "#2ecc71",  # 2. 그린
        "#e74c3c",  # 3. 레드
        "#f1c40f",  # 4. 옐로우
        "#9b59b6",  # 5. 퍼플
        "#e67e22",  # 6. 오렌지
        "#1abc9c",  # 7. 민트
        "#34495e",  # 8. 다크그레이
    ]

    # 중심점(Center)을 더 돋보이게 해줄 진한 색상 리스트
    center_colors = [
        "#1f3a60",  # 1. 딥블루
        "#145a32",  # 2. 딥그린
        "#78281f",  # 3. 딥레드
        "#b7950b",  # 4. 다크옐로우
        "#512e5f",  # 5. 딥퍼플
        "#7e5109",  # 6. 브라운오렌지
        "#0e6251",  # 7. 딥민트
        "#1c2833",  # 8. 블랙그레이
    ]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 2. 클러스터별로 루프를 돌며 점, 중심점, 원 그리기
    for i, cluster in enumerate(clusters):
        cx, cy = centers[i]["x"], centers[i]["y"]

        xs = [pt["x"] for pt in cluster]
        ys = [pt["y"] for pt in cluster]

        # 일반 점 그리기
        ax.scatter(
            xs,
            ys,
            color=colors[i],
            s=100,
            label=f"Cluster {i+1} Points",
            edgecolors="black",
            zorder=3,
        )

        # 중심점 그리기
        ax.scatter(
            cx,
            cy,
            color=center_colors[i],
            marker="X",
            s=300,
            label=f"Center {i+1}",
            edgecolors="black",
            zorder=4,
        )

        # 💡 [원 그리기 핵심] 중심점에서 가장 먼 점과의 거리를 구해서 반지름(radius)으로 설정
        max_dist = 0
        for pt in cluster:
            dist = ((pt["x"] - cx) ** 2 + (pt["y"] - cy) ** 2) ** 0.5
            if dist > max_dist:
                max_dist = dist

        # 원 객체 생성 (alpha로 투명도 조절, 생삭은 점과 매칭)
        circle = patches.Circle(
            (cx, cy),
            radius=max_dist,
            color=colors[i],
            alpha=0.15,
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
        ax.add_patch(circle)

    # 3. 그래프 스타일 설정
    ax.set_title("Clustering with Enclosing Circles", fontsize=14, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    # 원이 잘리지 않도록 범위를 조금 더 넓게 설정
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

    # 이미지 저장 후 출력 (savefig는 반드시 show 전에!)
    plt.savefig("cluster_circles.png", dpi=300, bbox_inches="tight")
