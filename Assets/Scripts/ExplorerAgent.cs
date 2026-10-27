using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.InputSystem;

public class ExplorerAgent : Agent
{
    private static PhysicsMaterial frictionlessMovementMaterial;

    [SerializeField] private Rigidbody parentRB;
    private LearnManager learnManager;

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

    [Header("Movement History Observation")]
    [Tooltip("直近の移動距離を集計するOnActionReceived呼び出し数。DecisionPeriod=5なら100で約20回の判断分です。")]
    [SerializeField, Min(1)] private int movementHistoryWindowSteps = 100;
    private const int VisitCountObservationCap = 3;
    private readonly Dictionary<Vector2Int, int> visitCountByCell =
        new Dictionary<Vector2Int, int>();
    private Vector2Int lastVisitCountCell;
    private bool hasLastVisitCountCell;
    private readonly Queue<Vector2> recentPlanarDisplacements = new Queue<Vector2>();
    private float recentPlanarTravelDistance;
    private Vector2 recentPlanarNetDisplacement;

    [Header("Vision")]
    [Tooltip("視界として情報を入手可能な距離。RayPerceptionSensor3Dとは別")]public float visionDistance = 15f;
    [Tooltip("視界として情報を入手可能な角度制限")]public float visionAngle = 90f;

    [Header("Stuck Diagnostics")]
    [Tooltip("停滞診断を有効化します。並列学習ではログが増えるため、必要なときだけ有効にしてください。")]
    [SerializeField] private bool enableStuckDiagnostics;
    [Tooltip("この回数分の行動履歴を使って、往復を含む停滞を判定します。")]
    [SerializeField, Min(2)] private int stuckDiagnosticWindowActionCount = 10;
    [Tooltip("履歴区間の正味移動がこの距離未満なら停滞候補とします。")]
    [SerializeField, Min(0.01f)] private float stuckDiagnosticNetDisplacementThreshold = 0.25f;

    //private Rigidbody rb;
    private HashSet<Vector2Int> visitedCells = new HashSet<Vector2Int>();
    private bool foundGoal = false;
    private bool canGoal = false;
    private bool episodeEnded;
    private int stepCount;
    public int maxStep = 5000;

    private InputSystem_Actions inputActions;
    private Vector2 moveInput;

    private readonly List<ContactPoint> collisionContactPoints = new List<ContactPoint>();
    private readonly Dictionary<Collider, List<Vector3>> wallContactNormals =
        new Dictionary<Collider, List<Vector3>>();

    private Transform goal;
    private Vector3 episodeSpawnPosition;
    private Vector3 episodeGoalPosition;

    public void BindToLearnManager(LearnManager manager)
    {
        learnManager = manager;
    }

    public bool TryGetVerticalSpawnAlignment(out float bodyWorldY, out float footWorldY)
    {
        bodyWorldY = parentRB != null ? parentRB.position.y : transform.position.y;
        footWorldY = bodyWorldY;

        if (parentRB == null)
        {
            return false;
        }

        Collider[] colliders = parentRB.GetComponentsInChildren<Collider>(true);
        bool foundBodyCollider = false;
        foreach (Collider bodyCollider in colliders)
        {
            if (bodyCollider == null || !bodyCollider.enabled || bodyCollider.isTrigger ||
                bodyCollider.attachedRigidbody != parentRB)
            {
                continue;
            }

            if (!foundBodyCollider || bodyCollider.bounds.min.y < footWorldY)
            {
                footWorldY = bodyCollider.bounds.min.y;
                foundBodyCollider = true;
            }
        }

        return foundBodyCollider;
    }

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
    private readonly List<StuckDiagnosticSample> recentStuckDiagnosticSamples =
        new List<StuckDiagnosticSample>();
    private bool stuckDiagnosticLogged;
    private bool hasPreviousDiagnosticAction;
    private Vector2 previousDiagnosticAction;
    private Vector3 previousRequestedVelocity;
    private Vector3 previousWallCorrectedVelocity;

    private struct StuckDiagnosticSample
    {
        public Vector3 Position { get; }
        public Vector3 Displacement { get; }
        public Vector2 Action { get; }
        public Vector3 RequestedVelocity { get; }
        public Vector3 WallCorrectedVelocity { get; }

        public StuckDiagnosticSample(
            Vector3 position,
            Vector3 displacement,
            Vector2 action,
            Vector3 requestedVelocity,
            Vector3 wallCorrectedVelocity)
        {
            Position = position;
            Displacement = displacement;
            Action = action;
            RequestedVelocity = requestedVelocity;
            WallCorrectedVelocity = wallCorrectedVelocity;
        }
    }

    // ゲーム開始時に呼ばれる
    void Awake()
    {
        inputActions = new InputSystem_Actions();

        inputActions.Player.Move.performed += ctx =>
           moveInput = ctx.ReadValue<Vector2>();

        inputActions.Player.Move.canceled += ctx =>
            moveInput = Vector2.zero;

        ConfigureFrictionlessMovement();

        //rb = GetComponent<Rigidbody>();
        //// 転倒防止: X,Z軸回転を固定
        //rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        //// 移動安定用に線形ドラグを設定
        //rb.linearDamping = 1.0f;
        inputActions.Player.Enable(); //これやんないとinputSystemが動かん
    }

    private void ConfigureFrictionlessMovement()
    {
        if (parentRB == null)
        {
            return;
        }

        if (frictionlessMovementMaterial == null)
        {
            frictionlessMovementMaterial = new PhysicsMaterial("Explorer Agent Frictionless")
            {
                staticFriction = 0f,
                dynamicFriction = 0f,
                frictionCombine = PhysicsMaterialCombine.Minimum,
                bounciness = 0f,
                bounceCombine = PhysicsMaterialCombine.Minimum
            };
        }

        Collider[] bodyColliders = parentRB.GetComponentsInChildren<Collider>(true);
        foreach (Collider bodyCollider in bodyColliders)
        {
            if (bodyCollider == null || !bodyCollider.enabled || bodyCollider.isTrigger ||
                bodyCollider.attachedRigidbody != parentRB)
            {
                continue;
            }

            bodyCollider.sharedMaterial = frictionlessMovementMaterial;
        }
    }

    void Update()
    {
        // フォールオフ判定：床下に落ちたら大きなペナルティで終了
        if (transform.position.y < -10f)
        {
            AddReward(-1f);
            EndThisEpisode(false);
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
        episodeEnded = false;
        wallContactNormals.Clear();
        ResetStuckDiagnostics();
        // ゴール到達可能になるまで遅延を挿入
        canGoal = false;
        StartCoroutine(InstantGoalLimit());

        // マップ再生成（Environmentリセット）
        if (learnManager == null)
        {
            Debug.LogError("ExplorerAgent must be spawned and registered by a LearnManager.", this);
            return;
        }

        GeneratedMap map = learnManager.BeginEpisode(this);
        episodeSpawnPosition = map.SpawnPosition;
        episodeGoalPosition = map.GoalPosition;
        goal = null;

        // 物理量リセット
        parentRB.linearVelocity = Vector3.zero;
        parentRB.angularVelocity = Vector3.zero;

        // エージェント位置・回転設定
        Vector3 spawnPosition = episodeSpawnPosition;
        if ((parentRB.constraints & RigidbodyConstraints.FreezePositionY) != 0)
        {
            spawnPosition.y = parentRB.position.y;
        }

        // Keep the configured vertical lock; generated maps align their floor to this height.
        parentRB.position = spawnPosition;

        transform.rotation = Quaternion.Euler(0, map.SpawnRotation, 0);

        // 探索セル履歴初期化
        visitedCells.Clear();
        visitCountByCell.Clear();
        hasLastVisitCountCell = false;
        memoryMap.Clear();
        RegisterVisited(transform.position);
        ResetMovementHistory();

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
        Vector3 delta = transform.position - lastPosition;

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
        AddVisitCountObservation(sensor);
        AddMovementHistoryObservation(sensor);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        UpdateMovementHistory();
        UpdateMemoryFromRaycasts();

        float x = actions.ContinuousActions[0];
        float z = actions.ContinuousActions[1];
        Vector2 rawAction = new Vector2(x, z);

        if (enableStuckDiagnostics)
        {
            UpdateStuckDiagnostics(rawAction);
        }
        else
        {
            ResetStuckDiagnostics();
        }



        actionXSum += x;
        actionZSum += z;
        actionSampleCount++;
        actionXAbsSum += Mathf.Abs(x);
        actionZAbsSum += Mathf.Abs(z);


        Vector3 moveDir = new Vector3(x, 0f, z);
        Vector3 requestedVelocity = Vector3.zero;
        Vector3 wallCorrectedVelocity = Vector3.zero;

        if (moveDir.sqrMagnitude > 0.001f)
        {
            moveDir.Normalize();

            requestedVelocity = moveDir * moveSpeed;
            wallCorrectedVelocity = SlideVelocityAlongWalls(requestedVelocity);

            // This is a dynamic Rigidbody: drive its horizontal velocity and let
            // the physics solver resolve contacts instead of moving its position.
            Vector3 velocity = parentRB.linearVelocity;
            velocity.x = wallCorrectedVelocity.x;
            velocity.z = wallCorrectedVelocity.z;
            parentRB.linearVelocity = velocity;

            transform.forward = moveDir;
        }
        else
        {
            Vector3 velocity = parentRB.linearVelocity;
            velocity.x = 0f;
            velocity.z = 0f;
            parentRB.linearVelocity = velocity;
        }

        if (enableStuckDiagnostics)
        {
            previousDiagnosticAction = rawAction;
            previousRequestedVelocity = requestedVelocity;
            previousWallCorrectedVelocity = wallCorrectedVelocity;
            hasPreviousDiagnosticAction = true;
        }

        // 3) 常時ステップペナルティ
        AddReward(stepPenalty);

        // 4) 新規セル探索報酬
        Vector2Int cell = WorldToCell(transform.position);
        bool firstVisit = visitedCells.Add(cell);
        RecordCellVisit(cell);
        if (firstVisit)
        {
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
        lastPosition = transform.position;
        // 最大ステップ到達でエピソード終了（失敗扱い、報酬追加なし）
        if (stepCount >= maxStep)
        {
            GoalDistanceRewardFoundedGoal();
            EndThisEpisode(false);
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
            AddReward((Vector3.Distance(episodeSpawnPosition, episodeGoalPosition) -
                Vector3.Distance(transform.position, episodeGoalPosition)) *
                goalDistanceRewardMultiplier);
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag("Wall"))
        {
            UpdateWallContactNormals(collision);
            AddReward(wallPenalty);
        }
    }

    private void OnCollisionStay(Collision collision)
    {
        if (collision.collider.CompareTag("Wall"))
        {
            UpdateWallContactNormals(collision);
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        wallContactNormals.Remove(collision.collider);
    }

    private void UpdateWallContactNormals(Collision collision)
    {
        Collider wallCollider = collision.collider;
        if (!wallContactNormals.TryGetValue(wallCollider, out List<Vector3> normals))
        {
            normals = new List<Vector3>();
            wallContactNormals.Add(wallCollider, normals);
        }

        normals.Clear();
        collisionContactPoints.Clear();
        collision.GetContacts(collisionContactPoints);

        Vector3 bodyCenter = parentRB.worldCenterOfMass;
        foreach (ContactPoint contactPoint in collisionContactPoints)
        {
            Vector3 normal = contactPoint.normal;
            normal.y = 0f;
            if (normal.sqrMagnitude <= 0.001f)
            {
                continue;
            }

            normal.Normalize();
            if (Vector3.Dot(normal, bodyCenter - contactPoint.point) < 0f)
            {
                normal = -normal;
            }

            normals.Add(normal);
        }

        if (normals.Count == 0)
        {
            wallContactNormals.Remove(wallCollider);
        }
    }

    private Vector3 SlideVelocityAlongWalls(Vector3 desiredVelocity)
    {
        // Remove only velocity directed into a wall. Tangential and outward
        // movement remain available, including when both action axes are used.
        for (int pass = 0; pass < 2; pass++)
        {
            foreach (List<Vector3> normals in wallContactNormals.Values)
            {
                foreach (Vector3 normal in normals)
                {
                    float velocityIntoWall = Vector3.Dot(desiredVelocity, normal);
                    if (velocityIntoWall < 0f)
                    {
                        desiredVelocity -= normal * velocityIntoWall;
                    }
                }
            }
        }

        return desiredVelocity;
    }

    private void UpdateStuckDiagnostics(Vector2 currentAction)
    {
        Vector3 currentPosition = transform.position;
        Vector3 displacement = currentPosition - lastPosition;
        if (hasPreviousDiagnosticAction)
        {
            recentStuckDiagnosticSamples.Add(new StuckDiagnosticSample(
                currentPosition,
                displacement,
                previousDiagnosticAction,
                previousRequestedVelocity,
                previousWallCorrectedVelocity));
        }

        int windowActionCount = Mathf.Max(2, stuckDiagnosticWindowActionCount);
        while (recentStuckDiagnosticSamples.Count > windowActionCount)
        {
            recentStuckDiagnosticSamples.RemoveAt(0);
        }

        if (recentStuckDiagnosticSamples.Count < windowActionCount)
        {
            return;
        }

        Vector3 windowStartPosition =
            recentStuckDiagnosticSamples[0].Position - recentStuckDiagnosticSamples[0].Displacement;
        Vector3 netDisplacement = recentStuckDiagnosticSamples[windowActionCount - 1].Position -
            windowStartPosition;
        float netPlanarDistance = new Vector2(netDisplacement.x, netDisplacement.z).magnitude;
        float totalPlanarTravel = 0f;
        foreach (StuckDiagnosticSample sample in recentStuckDiagnosticSamples)
        {
            totalPlanarTravel += new Vector2(sample.Displacement.x, sample.Displacement.z).magnitude;
        }

        if (netPlanarDistance >= Mathf.Max(0.01f, stuckDiagnosticNetDisplacementThreshold))
        {
            stuckDiagnosticLogged = false;
            return;
        }

        if (stuckDiagnosticLogged)
        {
            return;
        }

        Debug.LogWarning(
            $"[StuckDiagnostic] Agent={name}, step={stepCount}, " +
            $"windowActions={windowActionCount}, netPlanarDisplacement={netPlanarDistance:F3}m, " +
            $"totalPlanarTravel={totalPlanarTravel:F3}m, position={parentRB.position}, " +
            $"forward={transform.forward}, " +
            $"movementHistory={FormatStuckDiagnosticSamples()}, " +
            $"currentAction={FormatAction(currentAction)}, " +
            $"rigidbodyVelocity={FormatPlanarVelocity(parentRB.linearVelocity)}, " +
            $"foundGoal={foundGoal}, visitedCells={visitedCells.Count}, " +
            $"wallContacts={FormatWallContactNormals()}",
            this);

        stuckDiagnosticLogged = true;
    }

    private void ResetStuckDiagnostics()
    {
        recentStuckDiagnosticSamples.Clear();
        stuckDiagnosticLogged = false;
        hasPreviousDiagnosticAction = false;
        previousDiagnosticAction = Vector2.zero;
        previousRequestedVelocity = Vector3.zero;
        previousWallCorrectedVelocity = Vector3.zero;
    }

    private string FormatStuckDiagnosticSamples()
    {
        List<string> samples = new List<string>(recentStuckDiagnosticSamples.Count);
        foreach (StuckDiagnosticSample sample in recentStuckDiagnosticSamples)
        {
            samples.Add(
                $"a={FormatAction(sample.Action)} " +
                $"req={FormatPlanarVelocity(sample.RequestedVelocity)} " +
                $"slide={FormatPlanarVelocity(sample.WallCorrectedVelocity)} " +
                $"d=({sample.Displacement.x:F3},{sample.Displacement.z:F3})");
        }

        return string.Join(" | ", samples);
    }

    private string FormatWallContactNormals()
    {
        if (wallContactNormals.Count == 0)
        {
            return "none";
        }

        List<string> contacts = new List<string>();
        foreach (KeyValuePair<Collider, List<Vector3>> entry in wallContactNormals)
        {
            if (entry.Key == null || entry.Value.Count == 0)
            {
                continue;
            }

            List<string> normals = new List<string>();
            foreach (Vector3 normal in entry.Value)
            {
                normals.Add($"({normal.x:F2},{normal.z:F2})");
            }

            contacts.Add($"{entry.Key.name}:[{string.Join(",", normals)}]");
        }

        return contacts.Count == 0 ? "none" : string.Join(";", contacts);
    }

    private static string FormatAction(Vector2 action)
    {
        return $"({action.x:F3},{action.y:F3})";
    }

    private static string FormatPlanarVelocity(Vector3 velocity)
    {
        return $"({velocity.x:F3},{velocity.z:F3})";
    }

    private void OnTriggerEnter(Collider other)
    {
        if (!canGoal) return;
        if (other.CompareTag("Goal"))
        {
            AddReward(goalReward);
            Debug.Log("Goal! Episode reward: " + GetCumulativeReward());
            EndThisEpisode(true);
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

    private void EndThisEpisode(bool reachedGoal)
    {
        if (episodeEnded) return;
        episodeEnded = true;

        if (learnManager != null)
        {
            bool testRunComplete = learnManager.RecordEpisodeResult(this, reachedGoal);
            if (testRunComplete)
            {
                // Leave the final test episode visible in the scene without
                // starting an unscored episode after seed 1000.
                parentRB.linearVelocity = Vector3.zero;
                parentRB.angularVelocity = Vector3.zero;
                parentRB.isKinematic = true;
                enabled = false;
                return;
            }
        }

        // エピソード終了
        Debug.Log(GetCumulativeReward());
        EndEpisode();
    }

    private void RegisterVisited(Vector3 pos)
    {
        Vector2Int cell = WorldToCell(pos);

        visitedCells.Add(cell);
        visitCountByCell[cell] = 1;
        lastVisitCountCell = cell;
        hasLastVisitCountCell = true;

        memoryMap[cell] = MemoryCell.Empty;
    }

    private void RecordCellVisit(Vector2Int cell)
    {
        if (hasLastVisitCountCell && cell == lastVisitCountCell)
        {
            return;
        }

        visitCountByCell.TryGetValue(cell, out int currentVisitCount);
        visitCountByCell[cell] = Mathf.Min(
            currentVisitCount + 1,
            VisitCountObservationCap);
        lastVisitCountCell = cell;
        hasLastVisitCountCell = true;
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

    private void AddVisitCountObservation(VectorSensor sensor, int radius = 5)
    {
        Vector2Int center = WorldToCell(transform.position);

        for (int z = -radius; z <= radius; z++)
        {
            for (int x = -radius; x <= radius; x++)
            {
                Vector2Int cell = new Vector2Int(center.x + x, center.y + z);
                visitCountByCell.TryGetValue(cell, out int visitCount);
                sensor.AddObservation(
                    Mathf.Clamp01((float)visitCount / VisitCountObservationCap));
            }
        }
    }

    private void UpdateMovementHistory()
    {
        Vector3 displacement = transform.position - lastPosition;
        Vector2 planarDisplacement = new Vector2(displacement.x, displacement.z);
        recentPlanarDisplacements.Enqueue(planarDisplacement);
        recentPlanarTravelDistance += planarDisplacement.magnitude;
        recentPlanarNetDisplacement += planarDisplacement;

        int windowSteps = Mathf.Max(1, movementHistoryWindowSteps);
        while (recentPlanarDisplacements.Count > windowSteps)
        {
            Vector2 expiredDisplacement = recentPlanarDisplacements.Dequeue();
            recentPlanarTravelDistance -= expiredDisplacement.magnitude;
            recentPlanarNetDisplacement -= expiredDisplacement;
        }

        recentPlanarTravelDistance = Mathf.Max(0f, recentPlanarTravelDistance);
    }

    private void ResetMovementHistory()
    {
        recentPlanarDisplacements.Clear();
        recentPlanarTravelDistance = 0f;
        recentPlanarNetDisplacement = Vector2.zero;
    }

    private void AddMovementHistoryObservation(VectorSensor sensor)
    {
        int windowSteps = Mathf.Max(1, movementHistoryWindowSteps);
        float maximumWindowDistance = Mathf.Max(
            moveSpeed * Time.fixedDeltaTime * windowSteps,
            0.001f);
        float normalizedTravelDistance = Mathf.Clamp01(
            recentPlanarTravelDistance / maximumWindowDistance);
        float movementEfficiency = recentPlanarTravelDistance > 0.001f
            ? Mathf.Clamp01(recentPlanarNetDisplacement.magnitude / recentPlanarTravelDistance)
            : 0f;

        sensor.AddObservation(normalizedTravelDistance);
        sensor.AddObservation(movementEfficiency);
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
