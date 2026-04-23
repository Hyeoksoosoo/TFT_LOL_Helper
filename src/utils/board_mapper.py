class BoardMapper:
    def __init__(self):
        # 판정 범위 (박스 크기 70보다 약간 넉넉하게)
        self.THRESHOLD_DIST = 85
        
        # 1. 🟢 보드판 좌표 (직접 찍은 완벽한 좌표)
        self.BOARD_SLOTS = [
            # Row 0
            {'index': [0, 0], 'center': (1189, 592)},
            {'index': [0, 1], 'center': (1341, 597)},
            {'index': [0, 2], 'center': (1496, 596)},
            {'index': [0, 3], 'center': (1649, 599)},
            {'index': [0, 4], 'center': (1805, 597)},
            {'index': [0, 5], 'center': (1954, 596)},
            {'index': [0, 6], 'center': (2107, 594)},
            
            # Row 1
            {'index': [1, 0], 'center': (1251, 688)},
            {'index': [1, 1], 'center': (1411, 690)},
            {'index': [1, 2], 'center': (1570, 685)},
            {'index': [1, 3], 'center': (1726, 687)},
            {'index': [1, 4], 'center': (1885, 690)},
            {'index': [1, 5], 'center': (2043, 689)},
            {'index': [1, 6], 'center': (2200, 690)},
            
            # Row 2
            {'index': [2, 0], 'center': (1157, 792)},
            {'index': [2, 1], 'center': (1315, 790)},
            {'index': [2, 2], 'center': (1485, 793)},
            {'index': [2, 3], 'center': (1647, 790)},
            {'index': [2, 4], 'center': (1808, 794)},
            {'index': [2, 5], 'center': (1970, 790)},
            {'index': [2, 6], 'center': (2132, 792)},
            
            # Row 3
            {'index': [3, 0], 'center': (1216, 897)},
            {'index': [3, 1], 'center': (1385, 898)},
            {'index': [3, 2], 'center': (1559, 900)},
            {'index': [3, 3], 'center': (1726, 896)},
            {'index': [3, 4], 'center': (1897, 902)},
            {'index': [3, 5], 'center': (2063, 895)},
            {'index': [3, 6], 'center': (2234, 902)},
        ]
        
        # 2. 🔵 대기석 (수동 설정값 적용)
        self.BENCH_SLOTS = []
        
        bench_start_x = 1000  # 시작 위치
        bench_y = 1050        # 높이
        bench_step = 159      # 간격
        
        for i in range(9):
            bx = bench_start_x + (i * bench_step)
            self.BENCH_SLOTS.append({'index': [0, i], 'center': (bx, bench_y)})

    def get_location(self, unit_center_x, unit_center_y):
        import math
        min_dist = float('inf')
        best_match = None
        location_type = None

        for slot in self.BOARD_SLOTS:
            sx, sy = slot['center']
            dist = math.dist((unit_center_x, unit_center_y), (sx, sy))
            if dist < min_dist:
                min_dist = dist
                best_match = slot['index']
                location_type = 'board'

        for slot in self.BENCH_SLOTS:
            sx, sy = slot['center']
            dist = math.dist((unit_center_x, unit_center_y), (sx, sy))
            if dist < min_dist:
                min_dist = dist
                best_match = slot['index']
                location_type = 'bench'

        if min_dist > self.THRESHOLD_DIST: 
            return None, None
            
        return location_type, best_match