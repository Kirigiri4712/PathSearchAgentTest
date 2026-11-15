using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

public struct GeneratedMap
{
    public Vector3 SpawnPosition { get; }
    public Vector3 GoalPosition { get; }
    public float SpawnRotation { get; }

    public GeneratedMap(Vector3 spawnPosition, Vector3 goalPosition, float spawnRotation)
    {
        SpawnPosition = spawnPosition;
        GoalPosition = goalPosition;
        SpawnRotation = spawnRotation;
    }
}

public enum MapLayoutMode
{
    PathCorridor,
    RoomsAndCorridors
}

/// <summary>
/// ランダムマップ生成
/// ・1x1ブロックを配置して通路を作成
/// ・Goalは四つ角のどこか
/// ・PlayerはGoalの対角
/// ・経由地点をランダム配置
/// ・A*で Start → 経由地点 → Goal を接続
/// </summary>
public class MapGenerator : MonoBehaviour
{
    [Header("Objects")]
    public GameObject floorPrefab;
    public GameObject wallPrefab;
    public GameObject goalPrefab;

    private Transform mapRoot;
    private Mesh generatedFloorMesh;

    private bool[,] walkable;
    private System.Random random;
    private int width;
    private int height;
    private int waypointCount;
    private int lastSeed;
    private MapLayoutMode lastLayoutMode;

    private Vector2Int playerPos;
    private Vector2Int goalPos;

    public int Width => width;
    public int Height => height;
    public int WaypointCount => waypointCount;
    public int LastSeed => lastSeed;
    public MapLayoutMode LastLayoutMode => lastLayoutMode;

    private readonly Vector2Int[] directions =
    {
        Vector2Int.up,
        Vector2Int.down,
        Vector2Int.left,
        Vector2Int.right
    };

    public GeneratedMap Generate(
        int width,
        int height,
        int waypointCount,
        int seed,
        float playerLocalY,
        float floorSurfaceLocalY,
        MapLayoutMode layoutMode = MapLayoutMode.PathCorridor)
    {
        width = Mathf.Max(5, width);
        height = Mathf.Max(5, height);
        waypointCount = Mathf.Clamp(waypointCount, 0, (width - 4) * (height - 4));
        this.width = width;
        this.height = height;
        this.waypointCount = waypointCount;
        lastSeed = seed;
        lastLayoutMode = layoutMode;
        random = new System.Random(seed);

        if (mapRoot != null)
        {
            mapRoot.gameObject.SetActive(false);
            Destroy(mapRoot.gameObject);
        }
        if (generatedFloorMesh != null)
        {
            Destroy(generatedFloorMesh);
            generatedFloorMesh = null;
        }

        mapRoot = new GameObject("MapRoot").transform;
        mapRoot.SetParent(transform, false);

        walkable = new bool[width, height];

        // 最初は全部壁
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                walkable[x, y] = false;
            }
        }

        DetermineStartAndGoal();

        if (layoutMode == MapLayoutMode.RoomsAndCorridors)
        {
            GenerateRoomsAndCorridors();
        }
        else
        {
            GeneratePathCorridorLayout();
        }

        // スタートとゴール保証
        walkable[playerPos.x, playerPos.y] = true;
        walkable[goalPos.x, goalPos.y] = true;

        BuildMap(playerLocalY, floorSurfaceLocalY);

        GameObject goal = Instantiate(goalPrefab, mapRoot);
        goal.transform.localPosition = GridToLocal(goalPos, playerLocalY + 0.5f);
        goal.transform.localRotation = Quaternion.identity;

        return new GeneratedMap(
            transform.TransformPoint(GridToLocal(playerPos, playerLocalY)),
            goal.transform.position,
            (float)random.NextDouble() * 360f);
    }

    /// <summary>
    /// Returns a compact, stable text representation of the generated map.
    /// Rows are ordered from highest z to lowest z; columns increase with x.
    /// </summary>
    public string CreateDiagnosticMapSnapshot()
    {
        if (walkable == null)
        {
            return "Map unavailable: no map has been generated.";
        }

        StringBuilder snapshot = new StringBuilder();
        snapshot.AppendLine("Legend: #=wall, .=walkable, S=start, G=goal");
        snapshot.AppendLine("Rows: z decreases top-to-bottom; columns: x increases left-to-right.");
        for (int z = height - 1; z >= 0; z--)
        {
            snapshot.Append("z=").Append(z.ToString("D3")).Append(' ');
            for (int x = 0; x < width; x++)
            {
                char cell = walkable[x, z] ? '.' : '#';
                if (new Vector2Int(x, z) == playerPos)
                {
                    cell = 'S';
                }
                else if (new Vector2Int(x, z) == goalPos)
                {
                    cell = 'G';
                }

                snapshot.Append(cell);
            }

            if (z > 0)
            {
                snapshot.AppendLine();
            }
        }

        return snapshot.ToString();
    }

    private void GeneratePathCorridorLayout()
    {
        List<Vector2Int> waypoints = GenerateWaypoints();
        List<Vector2Int> route = new List<Vector2Int>();
        Vector2Int current = playerPos;

        foreach (Vector2Int waypoint in waypoints)
        {
            List<Vector2Int> path = FindPath(current, waypoint);
            if (path == null)
            {
                continue;
            }

            route.AddRange(path);
            current = waypoint;
        }

        List<Vector2Int> finalPath = FindPath(current, goalPos);
        if (finalPath != null)
        {
            route.AddRange(finalPath);
        }

        foreach (Vector2Int cell in route)
        {
            walkable[cell.x, cell.y] = true;
        }
    }

    private void GenerateRoomsAndCorridors()
    {
        int interiorWidth = width - 2;
        int interiorHeight = height - 2;
        int roomColumns = Mathf.Clamp(width / 10, 1, 5);
        int roomRows = Mathf.Clamp(height / 10, 1, 5);
        Vector2Int[,] roomCenters = new Vector2Int[roomColumns, roomRows];

        for (int roomX = 0; roomX < roomColumns; roomX++)
        {
            int regionMinX = 1 + roomX * interiorWidth / roomColumns;
            int regionMaxX = 1 + (roomX + 1) * interiorWidth / roomColumns - 1;
            int regionWidth = regionMaxX - regionMinX + 1;
            int targetRoomWidth = Mathf.Clamp(regionWidth * 2 / 3, 3, regionWidth);
            int actualRoomWidth = roomColumns == 1
                ? regionWidth
                : Mathf.Clamp(targetRoomWidth + random.Next(-1, 2), 1, regionWidth);
            int roomMinX = GetRoomStart(regionMinX, regionMaxX, actualRoomWidth, roomX, roomColumns);
            int roomMaxX = roomMinX + actualRoomWidth - 1;

            for (int roomZ = 0; roomZ < roomRows; roomZ++)
            {
                int regionMinZ = 1 + roomZ * interiorHeight / roomRows;
                int regionMaxZ = 1 + (roomZ + 1) * interiorHeight / roomRows - 1;
                int regionDepth = regionMaxZ - regionMinZ + 1;
                int targetRoomDepth = Mathf.Clamp(regionDepth * 2 / 3, 3, regionDepth);
                int actualRoomDepth = roomRows == 1
                    ? regionDepth
                    : Mathf.Clamp(targetRoomDepth + random.Next(-1, 2), 1, regionDepth);
                int roomMinZ = GetRoomStart(regionMinZ, regionMaxZ, actualRoomDepth, roomZ, roomRows);
                int roomMaxZ = roomMinZ + actualRoomDepth - 1;

                CarveRoom(roomMinX, roomMaxX, roomMinZ, roomMaxZ);
                roomCenters[roomX, roomZ] = new Vector2Int(
                    (roomMinX + roomMaxX) / 2,
                    (roomMinZ + roomMaxZ) / 2);
            }
        }

        // Connect every neighboring room pair. The grid remains connected while
        // its cycles create route choices around rooms and corridors.
        for (int roomX = 0; roomX < roomColumns; roomX++)
        {
            for (int roomZ = 0; roomZ < roomRows; roomZ++)
            {
                if (roomX + 1 < roomColumns)
                {
                    CarveCorridor(roomCenters[roomX, roomZ], roomCenters[roomX + 1, roomZ]);
                }

                if (roomZ + 1 < roomRows)
                {
                    CarveCorridor(roomCenters[roomX, roomZ], roomCenters[roomX, roomZ + 1]);
                }
            }
        }
    }

    private int GetRoomStart(int regionMin, int regionMax, int roomSize, int regionIndex, int regionCount)
    {
        if (regionCount == 1)
        {
            return regionMin;
        }

        if (regionIndex == 0)
        {
            return regionMin;
        }

        if (regionIndex == regionCount - 1)
        {
            return regionMax - roomSize + 1;
        }

        int availableOffset = Mathf.Max(0, regionMax - regionMin - roomSize + 1);
        return regionMin + random.Next(availableOffset + 1);
    }

    private void CarveRoom(int minX, int maxX, int minZ, int maxZ)
    {
        for (int x = minX; x <= maxX; x++)
        {
            for (int z = minZ; z <= maxZ; z++)
            {
                walkable[x, z] = true;
            }
        }
    }

    private void CarveCorridor(Vector2Int start, Vector2Int end)
    {
        Vector2Int corner = random.Next(0, 2) == 0
            ? new Vector2Int(end.x, start.y)
            : new Vector2Int(start.x, end.y);
        CarveLine(start, corner);
        CarveLine(corner, end);
    }

    private void CarveLine(Vector2Int start, Vector2Int end)
    {
        Vector2Int current = start;
        walkable[current.x, current.y] = true;
        while (current != end)
        {
            if (current.x != end.x)
            {
                current.x += current.x < end.x ? 1 : -1;
            }
            else
            {
                current.y += current.y < end.y ? 1 : -1;
            }

            if (InBounds(current))
            {
                walkable[current.x, current.y] = true;
            }
        }
    }

    private void DetermineStartAndGoal()
    {
        Vector2Int[] corners =
        {
            new Vector2Int(1, 1),
            new Vector2Int(width - 2, 1),
            new Vector2Int(1, height - 2),
            new Vector2Int(width - 2, height - 2)
        };

        int goalIndex = random.Next(0, 4);

        goalPos = corners[goalIndex];

        // 対角
        switch (goalIndex)
        {
            case 0:
                playerPos = corners[3];
                break;
            case 1:
                playerPos = corners[2];
                break;
            case 2:
                playerPos = corners[1];
                break;
            default:
                playerPos = corners[0];
                break;
        }
    }

    private List<Vector2Int> GenerateWaypoints()
    {
        List<Vector2Int> result = new List<Vector2Int>();

        int attempts = 0;

        while (result.Count < waypointCount && attempts < 1000)
        {
            attempts++;

            Vector2Int pos = new Vector2Int(
                random.Next(2, width - 2),
                random.Next(2, height - 2)
            );

            if (pos == playerPos || pos == goalPos)
                continue;

            if (result.Contains(pos))
                continue;

            result.Add(pos);
        }

        return result;
    }

    private void BuildMap(float playerLocalY, float floorSurfaceLocalY)
    {
        float floorMeshBottomOffset = GetFloorMeshBottomOffset();
        List<Vector3> floorVertices = new List<Vector3>();
        List<int> floorTriangles = new List<int>();

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                bool isWalkable = walkable[x, y];
                GameObject prefab = isWalkable ? floorPrefab : wallPrefab;

                GameObject tile = Instantiate(prefab, mapRoot);
                float tileLocalY = isWalkable
                    ? floorSurfaceLocalY - floorMeshBottomOffset
                    : playerLocalY;
                tile.transform.localPosition = new Vector3(x, tileLocalY, y);
                tile.transform.localRotation = Quaternion.identity;

                Collider tileCollider = tile.GetComponent<Collider>();
                if (tileCollider != null)
                {
                    tileCollider.enabled = false;
                }

                if (isWalkable)
                {
                    AddFloorCellGeometry(floorVertices, floorTriangles, x, y, tileLocalY);
                }
            }
        }

        BuildFloorCollider(floorVertices, floorTriangles);
        BuildMergedWallColliders(playerLocalY);
    }

    private void AddFloorCellGeometry(
        List<Vector3> vertices,
        List<int> triangles,
        int x,
        int y,
        float tileLocalY)
    {
        MeshFilter floorMeshFilter = floorPrefab.GetComponent<MeshFilter>();
        Vector3 meshSize = floorMeshFilter != null && floorMeshFilter.sharedMesh != null
            ? Vector3.Scale(floorMeshFilter.sharedMesh.bounds.size, floorPrefab.transform.localScale)
            : Vector3.one;
        float halfWidth = meshSize.x * 0.5f;
        float halfDepth = meshSize.z * 0.5f;
        float surfaceY = tileLocalY;
        if (floorMeshFilter != null && floorMeshFilter.sharedMesh != null)
        {
            surfaceY += floorMeshFilter.sharedMesh.bounds.max.y * floorPrefab.transform.localScale.y;
        }

        int firstVertex = vertices.Count;
        vertices.Add(new Vector3(x - halfWidth, surfaceY, y - halfDepth));
        vertices.Add(new Vector3(x - halfWidth, surfaceY, y + halfDepth));
        vertices.Add(new Vector3(x + halfWidth, surfaceY, y + halfDepth));
        vertices.Add(new Vector3(x + halfWidth, surfaceY, y - halfDepth));
        triangles.Add(firstVertex);
        triangles.Add(firstVertex + 1);
        triangles.Add(firstVertex + 2);
        triangles.Add(firstVertex);
        triangles.Add(firstVertex + 2);
        triangles.Add(firstVertex + 3);
    }

    private void BuildFloorCollider(List<Vector3> vertices, List<int> triangles)
    {
        if (vertices.Count == 0)
        {
            return;
        }

        generatedFloorMesh = new Mesh { name = "GeneratedFloorCollider" };
        if (vertices.Count > 65535)
        {
            generatedFloorMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        }
        generatedFloorMesh.SetVertices(vertices);
        generatedFloorMesh.SetTriangles(triangles, 0);
        generatedFloorMesh.RecalculateBounds();

        GameObject floorColliderObject = new GameObject("FloorCollider");
        floorColliderObject.transform.SetParent(mapRoot, false);
        MeshCollider floorCollider = floorColliderObject.AddComponent<MeshCollider>();
        floorCollider.sharedMesh = generatedFloorMesh;
    }

    private void BuildMergedWallColliders(float playerLocalY)
    {
        bool[,] merged = new bool[width, height];
        BoxCollider sourceCollider = wallPrefab.GetComponent<BoxCollider>();
        Vector3 wallScale = wallPrefab.transform.localScale;
        Vector3 sourceSize = sourceCollider != null ? sourceCollider.size : Vector3.one;
        Vector3 sourceCenter = sourceCollider != null ? sourceCollider.center : Vector3.zero;
        float cellWidth = Mathf.Abs(wallScale.x * sourceSize.x);
        float cellDepth = Mathf.Abs(wallScale.z * sourceSize.z);
        float wallHeight = Mathf.Abs(wallScale.y * sourceSize.y);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (walkable[x, y] || merged[x, y])
                {
                    continue;
                }

                int runWidth = 1;
                while (x + runWidth < width && !walkable[x + runWidth, y] && !merged[x + runWidth, y])
                {
                    runWidth++;
                }

                int runDepth = 1;
                bool canExtend = true;
                while (y + runDepth < height && canExtend)
                {
                    for (int offsetX = 0; offsetX < runWidth; offsetX++)
                    {
                        if (walkable[x + offsetX, y + runDepth] || merged[x + offsetX, y + runDepth])
                        {
                            canExtend = false;
                            break;
                        }
                    }
                    if (canExtend)
                    {
                        runDepth++;
                    }
                }

                for (int offsetX = 0; offsetX < runWidth; offsetX++)
                {
                    for (int offsetY = 0; offsetY < runDepth; offsetY++)
                    {
                        merged[x + offsetX, y + offsetY] = true;
                    }
                }

                GameObject wallColliderObject = new GameObject("WallCollider");
                wallColliderObject.tag = "Wall";
                wallColliderObject.transform.SetParent(mapRoot, false);
                wallColliderObject.transform.localPosition = new Vector3(
                    x + (runWidth - 1) * 0.5f + sourceCenter.x * wallScale.x,
                    playerLocalY + sourceCenter.y * wallScale.y,
                    y + (runDepth - 1) * 0.5f + sourceCenter.z * wallScale.z);
                BoxCollider wallCollider = wallColliderObject.AddComponent<BoxCollider>();
                wallCollider.size = new Vector3(runWidth * cellWidth, wallHeight, runDepth * cellDepth);
            }
        }
    }

    private void OnDestroy()
    {
        if (generatedFloorMesh != null)
        {
            Destroy(generatedFloorMesh);
        }
    }

    private float GetFloorMeshBottomOffset()
    {
        MeshFilter floorMeshFilter = floorPrefab.GetComponent<MeshFilter>();
        if (floorMeshFilter != null && floorMeshFilter.sharedMesh != null)
        {
            return floorMeshFilter.sharedMesh.bounds.min.y * floorPrefab.transform.localScale.y;
        }

        MeshCollider floorMeshCollider = floorPrefab.GetComponent<MeshCollider>();
        if (floorMeshCollider != null && floorMeshCollider.sharedMesh != null)
        {
            return floorMeshCollider.sharedMesh.bounds.min.y * floorPrefab.transform.localScale.y;
        }

        return 0f;
    }

    private Vector3 GridToLocal(Vector2Int pos, float localY)
    {
        return new Vector3(pos.x, localY, pos.y);
    }

    // =========================
    // A*
    // =========================

    private class Node
    {
        public Vector2Int pos;
        public Node parent;

        public int gCost;
        public int hCost;

        public int FCost => gCost + hCost;

        public Node(Vector2Int p)
        {
            pos = p;
        }
    }

    private List<Vector2Int> FindPath(Vector2Int start, Vector2Int end)
    {
        List<Node> openList = new List<Node>();
        HashSet<Vector2Int> closedSet = new HashSet<Vector2Int>();

        Node startNode = new Node(start)
        {
            gCost = 0,
            hCost = Heuristic(start, end)
        };

        openList.Add(startNode);

        while (openList.Count > 0)
        {
            Node current = openList[0];

            for (int i = 1; i < openList.Count; i++)
            {
                if (openList[i].FCost < current.FCost ||
                    (openList[i].FCost == current.FCost &&
                     openList[i].hCost < current.hCost))
                {
                    current = openList[i];
                }
            }

            openList.Remove(current);
            closedSet.Add(current.pos);

            if (current.pos == end)
            {
                return RetracePath(current);
            }

            foreach (var dir in directions)
            {
                Vector2Int nextPos = current.pos + dir;

                if (!InBounds(nextPos))
                    continue;

                if (closedSet.Contains(nextPos))
                    continue;

                int newCost = current.gCost + 1;

                Node existing =
                    openList.Find(n => n.pos == nextPos);

                if (existing == null)
                {
                    Node node = new Node(nextPos)
                    {
                        gCost = newCost,
                        hCost = Heuristic(nextPos, end),
                        parent = current
                    };

                    openList.Add(node);
                }
                else if (newCost < existing.gCost)
                {
                    existing.gCost = newCost;
                    existing.parent = current;
                }
            }
        }

        return null;
    }

    private List<Vector2Int> RetracePath(Node endNode)
    {
        List<Vector2Int> path = new List<Vector2Int>();

        Node current = endNode;

        while (current != null)
        {
            path.Add(current.pos);
            current = current.parent;
        }

        path.Reverse();

        return path;
    }

    private int Heuristic(Vector2Int a, Vector2Int b)
    {
        return Mathf.Abs(a.x - b.x) + Mathf.Abs(a.y - b.y);
    }

    private bool InBounds(Vector2Int pos)
    {
        return pos.x >= 0 &&
               pos.y >= 0 &&
               pos.x < width &&
               pos.y < height;
    }

}
