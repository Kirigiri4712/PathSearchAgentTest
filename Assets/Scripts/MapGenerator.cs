using System;
using System.Collections.Generic;
using UnityEngine;

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
    [Header("Map Size")]
    public int width = 40;
    public int height = 40;

    [Header("Objects")]
    public GameObject floorPrefab;
    public GameObject wallPrefab;
    public GameObject playerPrefab;
    public GameObject goalPrefab;

    private Transform mapRoot;

    [Header("Generation")]
    public int waypointCount = 4;
    public int randomSeed = 0;

    private bool[,] walkable;

    private Vector2Int playerPos;
    private Vector2Int goalPos;

    private readonly Vector2Int[] directions =
    {
        Vector2Int.up,
        Vector2Int.down,
        Vector2Int.left,
        Vector2Int.right
    };

    private void Start()
    {
        AgentSingleton.instance.mapGenerator = this;
        //Generate();
    }

    public void Generate()
    {
        if (mapRoot != null)
        {
            Destroy(mapRoot.gameObject);
        }

        mapRoot = new GameObject("MapRoot").transform;

        if (randomSeed != 0)
            UnityEngine.Random.InitState(randomSeed);

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

        List<Vector2Int> waypoints = GenerateWaypoints();

        List<Vector2Int> route = new List<Vector2Int>();

        Vector2Int current = playerPos;

        // 経由地点へ接続
        foreach (var wp in waypoints)
        {
            List<Vector2Int> path = FindPath(current, wp);

            if (path != null)
            {
                route.AddRange(path);
                current = wp;
            }
        }

        // Goalへ接続
        List<Vector2Int> finalPath = FindPath(current, goalPos);

        if (finalPath != null)
        {
            route.AddRange(finalPath);
        }

        // 通路化
        foreach (var p in route)
        {
            walkable[p.x, p.y] = true;
        }

        // スタートとゴール保証
        walkable[playerPos.x, playerPos.y] = true;
        walkable[goalPos.x, goalPos.y] = true;

        BuildMap();

        AgentSingleton.instance.SpawnPos = GridToWorld(playerPos);
        Instantiate(goalPrefab, GridToWorld(goalPos), Quaternion.identity, mapRoot);
        AgentSingleton.instance.GoalPos = GridToWorld(goalPos);
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

        int goalIndex = UnityEngine.Random.Range(0, 4);

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
                UnityEngine.Random.Range(2, width - 2),
                UnityEngine.Random.Range(2, height - 2)
            );

            if (pos == playerPos || pos == goalPos)
                continue;

            if (result.Contains(pos))
                continue;

            result.Add(pos);
        }

        return result;
    }

    private void BuildMap()
    {
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                GameObject prefab = walkable[x, y]
                    ? floorPrefab
                    : wallPrefab;

                Instantiate(prefab, new Vector3(x, 0, y), Quaternion.identity, mapRoot);
            }
        }
    }

    private Vector3 GridToWorld(Vector2Int pos)
    {
        return new Vector3(pos.x, 0.5f, pos.y);
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

    public void ResetMap()
    {
        Generate();
    }
}