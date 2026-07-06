using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.InputSystem;

public class ExplorerAgent : Agent
{
    [SerializeField] private Rigidbody parentRB;

    [Header("Movement")]
    public float moveSpeed = 4f;           // 前後移動の速度

    [Header("Rewards")]
    [Tooltip("ステップごとのペナルティ")]public float stepPenalty = -0.0005f;   // ステップごとのペナルティ
    [Tooltip("ゴール到達報酬")]public float goalReward = 10f;         // ゴール到達報酬
    [Tooltip("ゴール視認報酬（縮小）")]public float goalVisibleReward = 0.2f; // ゴール視認報酬（縮小）
    [Tooltip("ゴール視認後のゴール距離報酬")]public float goalDistanceRewardMultiplier = 0.0001f; //ゴール視認後のゴール距離報酬
    [Header("Action Rewards")]
    [Tooltip("新規セル到達報酬")]public float exploreReward = 0.02f;    // 新規セル到達報酬
    [Tooltip("ゴール視認後の新規セル報酬")]public float exploreRewardAfterGoalVisible = 0.005f; // ゴール視認後の新規セル報酬
    [Tooltip("前方開放ボーナス")]public float forwardOpenBonus = 0.005f;   // 前方開放ボーナス
    [Tooltip("前方壁ペナルティ")]public float forwardWallPenalty = -0.005f;  // 前方壁ペナルティ
    [Tooltip("壁衝突ペナルティ")]public float wallPenalty = -0.01f;  // 壁衝突ペナルティ
    [Tooltip("回転行動ペナルティ（必要なら小さく設定可）")]public float rotatePenalty = 0f;      // 回転行動ペナルティ（必要なら小さく設定可）

    [Header("Exploration")]
    [Tooltip("探索報酬対象セルサイズ（大きめにしてノイズ軽減）")]public float visitedCellSize = 2.5f; // 探索報酬対象セルサイズ（大きめにしてノイズ軽減）

    [Header("Vision")]
    [Tooltip("視界として情報を入手可能な距離。RayPerceptionSensor3Dとは別")]public float visionDistance = 15f;
    [Tooltip("視界として情報を入手可能な角度制限")]public float visionAngle = 90f;

    //private Rigidbody rb;
    private HashSet<Vector2Int> visitedCells = new HashSet<Vector2Int>();
    private bool foundGoal = false;
    private bool canGoal = false;
    private int stepCount;
    public int maxStep = 5000;

    private InputSystem_Actions inputActions;
    private Vector2 moveInput;

    private Transform goal;

    //探索済みマップ
    private enum MemoryCell
    {
        Unknown = 0,
        Empty = 1,
        Wall = 2
    }
    private Dictionary<Vector2Int, MemoryCell> memoryMap = new Dictionary<Vector2Int, MemoryCell>();



    //デバッグ
    private float actionXSum;
    private float actionZSum;
    private int actionSampleCount;
    private float actionXAbsSum;
    private float actionZAbsSum;

    // ゲーム開始時に呼ばれる
    void Awake()
    {
        inputActions = new InputSystem_Actions();

        inputActions.Player.Move.performed += ctx =>
           moveInput = ctx.ReadValue<Vector2>();

        inputActions.Player.Move.canceled += ctx =>
            moveInput = Vector2.zero;

        //rb = GetComponent<Rigidbody>();
        //// 転倒防止: X,Z軸回転を固定
        //rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        //// 移動安定用に線形ドラグを設定
        //rb.linearDamping = 1.0f;
        inputActions.Player.Enable(); //これやんないとinputSystemが動かん
    }
    void Update()
    {
        // フォールオフ判定：床下に落ちたら大きなペナルティで終了
        if (transform.position.y < -10f)
        {
            AddReward(-1f);
            EndThisEpisode();
        }
    }

    /// <summary>
    /// ゴール視認判定：ゴールタグがレイに映ると報酬
    /// </summary>
    private void CheckGoalVisible()
    {
        if (foundGoal) return;

        Collider[] hits = Physics.OverlapSphere(transform.position, visionDistance);
        foreach (Collider hit in hits)
        {
            if (!hit.CompareTag("Goal")) continue;
            Vector3 dir = (hit.transform.position - transform.position).normalized;
            float angle = Vector3.Angle(transform.forward, dir);
            if (angle > visionAngle * 0.5f) continue;
            // 壁で遮られていないか確認
            if (Physics.Raycast(transform.position + Vector3.up * 0.5f, dir, out RaycastHit rayHit, visionDistance))
            {
                if (rayHit.collider.CompareTag("Goal"))
                {
                    goal = rayHit.collider.transform;
                    foundGoal = true;
                    AddReward(goalVisibleReward);
                    Debug.Log("Goal Found!");
                }
            }
        }
    }

    public override void OnEpisodeBegin()
    {
        // ゴール到達可能になるまで遅延を挿入
        canGoal = false;
        StartCoroutine(InstantGoalLimit());

        // マップ再生成（Environmentリセット）
        MapGenerator mapGen = AgentSingleton.instance.mapGenerator;
        if (mapGen != null) mapGen.Generate();

        // 物理量リセット
        parentRB.linearVelocity = Vector3.zero;
        parentRB.angularVelocity = Vector3.zero;

        // エージェント位置・回転設定
        Vector3 spawn = AgentSingleton.instance.SpawnPos;
        transform.position = spawn + Vector3.up * 0.5f;
        transform.rotation = Quaternion.Euler(0, Random.Range(0f, 360f), 0);

        // 探索セル履歴初期化
        visitedCells.Clear();
        memoryMap.Clear();
        RegisterVisited(transform.position);

        // ループ用変数リセット
        stepCount = 0;
        foundGoal = false;
        lastPosition = transform.position;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 向き
        sensor.AddObservation(transform.forward.x);
        sensor.AddObservation(transform.forward.z);

        // 速度
        Vector3 delta =transform.position - lastPosition;

        sensor.AddObservation(delta.x);
        sensor.AddObservation(delta.z);

        // ゴール発見済みか
        sensor.AddObservation(foundGoal ? 1f : 0f);

        // ゴール発見後のみ方向を教える
        if (foundGoal)
        {
            Vector3 dir =
                transform.InverseTransformDirection(
                    goal.position - transform.position);

            dir.Normalize();

            sensor.AddObservation(dir.x);
            sensor.AddObservation(dir.z);

            sensor.AddObservation(
                Mathf.Clamp01(
                    Vector3.Distance(
                        transform.position,
                        goal.position
                    ) / 100f));
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // 探索進捗
        sensor.AddObservation(visitedCells.Count / 1000f);

        AddMemoryObservation(sensor);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        UpdateMemoryFromRaycasts();

        float x = actions.ContinuousActions[0];
        float z = actions.ContinuousActions[1];



        actionXSum += x;
        actionZSum += z;
        actionSampleCount++;
        actionXAbsSum += Mathf.Abs(x);
        actionZAbsSum += Mathf.Abs(z);


        Vector3 moveDir = new Vector3(x, 0f, z);

        if (moveDir.sqrMagnitude > 0.001f)
        {
            moveDir.Normalize();

            parentRB.MovePosition(parentRB.position + moveDir * moveSpeed * Time.fixedDeltaTime);

            transform.forward = moveDir;
        }
        else
        {
            parentRB.linearVelocity = Vector3.zero;
        }

        // 3) 常時ステップペナルティ
        AddReward(stepPenalty);

        // 4) 新規セル探索報酬
        Vector2Int cell = WorldToCell(transform.position);
        if (!visitedCells.Contains(cell))
        {
            visitedCells.Add(cell);
            AddReward(foundGoal ? exploreRewardAfterGoalVisible : exploreReward);
        }

        // 5) 前方壁/通路ボーナス
        Vector3 rayStart = transform.position + Vector3.up * 0.5f;
        Vector3 rayDir = transform.forward;
        float checkDist = 2f;
        bool hitWall = Physics.Raycast(rayStart, rayDir, out RaycastHit hit, checkDist);
        // レイ可視化（デバッグ用）
        Debug.DrawRay(rayStart, rayDir * checkDist, hitWall ? Color.red : Color.green);
        if (hitWall)
        {
            AddReward(forwardWallPenalty);
        }
        else
        {
            AddReward(forwardOpenBonus);
        }

        // 6) ゴール視認判定
        CheckGoalVisible();

        if (actionSampleCount >= 1000)
        {
            Debug.Log(
                $"Avg X={actionXSum / actionSampleCount:F3} " +
                $"Z={actionZSum / actionSampleCount:F3} " +
                $"|X|={actionXAbsSum / actionSampleCount:F3} " +
                $"|Z|={actionZAbsSum / actionSampleCount:F3}"
            );

            actionXSum = 0f;
            actionZSum = 0f;
            actionSampleCount = 0;
        }

        stepCount++;
        // 最大ステップ到達でエピソード終了（失敗扱い、報酬追加なし）
        if (stepCount >= maxStep)
        {
            GoalDistanceRewardFoundedGoal();
            EndThisEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var c = actionsOut.ContinuousActions;

        c[0] = moveInput.x; // A,D
        c[1] = moveInput.y; // W,S
        //Debug.Log(moveInput.x + " / " +  moveInput.y);
    }

    /// <summary>
    /// ゴールが見つかっている場合による距離報酬
    /// </summary>
    private void GoalDistanceRewardFoundedGoal()
    {
        if(foundGoal)
        {
            var ins = AgentSingleton.instance;
            if (ins != null)
            {
                AddReward((Vector3.Distance(ins.SpawnPos, ins.GoalPos) - Vector3.Distance(transform.position, ins.GoalPos)) * goalDistanceRewardMultiplier);
            }
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag("Wall"))
        {
            AddReward(wallPenalty);
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (!canGoal) return;
        if (other.CompareTag("Goal"))
        {
            AddReward(goalReward);
            Debug.Log("Goal! Episode reward: " + GetCumulativeReward());
            EndThisEpisode();
        }
    }

    private Vector3 lastPosition;
    // ゴール到達許可遅延コルーチン
    private IEnumerator InstantGoalLimit()
    {
        canGoal = false;
        yield return new WaitForSeconds(0.5f);
        canGoal = true;
    }

    private void EndThisEpisode()
    {
        // エピソード終了
        Debug.Log(GetCumulativeReward());
        EndEpisode();
    }

    private void RegisterVisited(Vector3 pos)
    {
        Vector2Int cell = WorldToCell(pos);

        visitedCells.Add(cell);

        memoryMap[cell] = MemoryCell.Empty;
    }

    private Vector2Int WorldToCell(Vector3 pos)
    {
        return new Vector2Int(
            Mathf.FloorToInt(pos.x / visitedCellSize),
            Mathf.FloorToInt(pos.z / visitedCellSize)
        );
    }

    private void UpdateMemoryFromRaycasts()
    {
        float rayLength = 15f;

        Vector3[] directions =
        {
        transform.forward,
        (transform.forward + transform.right).normalized,
        transform.right,
        (-transform.forward + transform.right).normalized,
        -transform.forward,
        (-transform.forward - transform.right).normalized,
        -transform.right,
        (transform.forward - transform.right).normalized
    };

        Vector3 start = transform.position + Vector3.up * 0.5f;

        foreach (var dir in directions)
        {
            if (Physics.Raycast(start, dir, out RaycastHit hit, rayLength))
            {
                if (hit.collider.CompareTag("Wall"))
                {
                    Vector2Int wallCell =
                        WorldToCell(hit.point);

                    memoryMap[wallCell] =
                        MemoryCell.Wall;
                }
            }
        }
    }

    private void AddMemoryObservation(
    VectorSensor sensor,
    int radius = 5)
    {
        Vector2Int center =
            WorldToCell(transform.position);

        for (int z = -radius; z <= radius; z++)
        {
            for (int x = -radius; x <= radius; x++)
            {
                Vector2Int cell =
                    new Vector2Int(
                        center.x + x,
                        center.y + z);

                if (!memoryMap.TryGetValue(
                        cell,
                        out MemoryCell state))
                {
                    sensor.AddObservation(-1f);
                    continue;
                }

                switch (state)
                {
                    case MemoryCell.Empty:
                        sensor.AddObservation(0f);
                        break;

                    case MemoryCell.Wall:
                        sensor.AddObservation(1f);
                        break;

                    default:
                        sensor.AddObservation(-1f);
                        break;
                }
            }
        }
    }

    // デバッグ：訪問セルを描画（エディタ画面で可視化）
    private void OnDrawGizmos()
    {
        if (visitedCells == null) return;
        Gizmos.color = Color.blue;
        foreach (Vector2Int cell in visitedCells)
        {
            float x = cell.x * visitedCellSize + visitedCellSize * 0.5f;
            float z = cell.y * visitedCellSize + visitedCellSize * 0.5f;
            Vector3 center = new Vector3(x, 0.01f, z);
            Gizmos.DrawCube(center, new Vector3(visitedCellSize, 0.02f, visitedCellSize));
        }
    }
}
