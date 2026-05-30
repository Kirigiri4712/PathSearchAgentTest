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
    public float stepPenalty = -0.0005f;   // ステップごとのペナルティ
    public float goalReward = 10f;         // ゴール到達報酬
    public float goalVisibleReward = 0.2f; // ゴール視認報酬（縮小）
    public float goalDistanceRewardMultiplier = 0.0001f; //ゴール視認後のゴール距離報酬
    [Header("Action Rewards")]
    public float exploreReward = 0.02f;                // 新規セル到達報酬
    public float exploreRewardAfterGoalVisible = 0.005f; // ゴール視認後の新規セル報酬
    public float forwardOpenBonus = 0.005f;            // 前方開放ボーナス
    public float forwardWallPenalty = -0.005f;         // 前方壁ペナルティ
    public float wallPenalty = -0.01f;                 // 壁衝突ペナルティ
    public float rotatePenalty = 0f;                   // 回転行動ペナルティ（必要なら小さく設定可）

    [Header("Exploration")]
    public float visitedCellSize = 2.5f; // 探索報酬対象セルサイズ（大きめにしてノイズ軽減）

    [Header("Vision")]
    public float visionDistance = 15f;
    public float visionAngle = 90f;

    //private Rigidbody rb;
    private HashSet<Vector2Int> visitedCells = new HashSet<Vector2Int>();
    private bool foundGoal = false;
    private bool canGoal = false;
    private int stepCount;
    public int maxStep = 5000;

    private InputSystem_Actions inputActions;
    private Vector2 moveInput;

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
        RegisterVisited(transform.position);

        // ループ用変数リセット
        stepCount = 0;
        foundGoal = false;
        lastPosition = transform.position;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // ローカル座標系での速度 (x, z)
        Vector3 localVel = transform.InverseTransformDirection(parentRB.linearVelocity);
        sensor.AddObservation(localVel.x);
        sensor.AddObservation(localVel.z);
        // (必要なら向き情報等を追加)
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float x = actions.ContinuousActions[0];
        float z = actions.ContinuousActions[1];

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

        stepCount++;
        // 最大ステップ到達でエピソード終了（失敗扱い、報酬追加なし）
        if (stepCount >= maxStep)
        {
            if (foundGoal)
            {
                var ins = AgentSingleton.instance;
                if (ins != null)
                {
                    AddReward((Vector3.Distance(ins.SpawnPos, ins.GoalPos) - Vector3.Distance(transform.position, ins.GoalPos)) * goalDistanceRewardMultiplier);
                }
                EndThisEpisode();
            }
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var c = actionsOut.ContinuousActions;

        c[0] = moveInput.x; // A,D
        c[1] = moveInput.y; // W,S
        //Debug.Log(moveInput.x + " / " +  moveInput.y);
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
        yield return new WaitForSeconds(1);
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
        visitedCells.Add(WorldToCell(pos));
    }

    private Vector2Int WorldToCell(Vector3 pos)
    {
        return new Vector2Int(
            Mathf.FloorToInt(pos.x / visitedCellSize),
            Mathf.FloorToInt(pos.z / visitedCellSize)
        );
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
